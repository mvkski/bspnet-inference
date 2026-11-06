"""Official BSP-Net network classes, unmodified; see THIRD_PARTY.md."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class generator(nn.Module):
	def __init__(self, phase, p_dim, c_dim):
		super(generator, self).__init__()
		self.phase = phase
		self.p_dim = p_dim
		self.c_dim = c_dim
		convex_layer_weights = torch.zeros((self.p_dim, self.c_dim))
		concave_layer_weights = torch.zeros((self.c_dim, 1))
		self.convex_layer_weights = nn.Parameter(convex_layer_weights)
		self.concave_layer_weights = nn.Parameter(concave_layer_weights)
		nn.init.normal_(self.convex_layer_weights, mean=0.0, std=0.02)
		nn.init.normal_(self.concave_layer_weights, mean=1e-5, std=0.02)

	def forward(self, points, plane_m, convex_mask=None, is_training=False):
		if self.phase==0:
			#level 1
			h1 = torch.matmul(points, plane_m)
			h1 = torch.clamp(h1, min=0)

			#level 2
			h2 = torch.matmul(h1, self.convex_layer_weights)
			h2 = torch.clamp(1-h2, min=0, max=1)

			#level 3
			h3 = torch.matmul(h2, self.concave_layer_weights)
			h3 = torch.clamp(h3, min=0, max=1)

			return h2,h3
		elif self.phase==1 or self.phase==2:
			#level 1
			h1 = torch.matmul(points, plane_m)
			h1 = torch.clamp(h1, min=0)

			#level 2
			h2 = torch.matmul(h1, (self.convex_layer_weights>0.01).float())

			#level 3
			if convex_mask is None:
				h3 = torch.min(h2, dim=2, keepdim=True)[0]
			else:
				h3 = torch.min(h2+convex_mask, dim=2, keepdim=True)[0]

			return h2,h3
		elif self.phase==3 or self.phase==4:
			#level 1
			h1 = torch.matmul(points, plane_m)
			h1 = torch.clamp(h1, min=0)

			#level 2
			h2 = torch.matmul(h1, self.convex_layer_weights)

			#level 3
			if convex_mask is None:
				h3 = torch.min(h2, dim=2, keepdim=True)[0]
			else:
				h3 = torch.min(h2+convex_mask, dim=2, keepdim=True)[0]

			return h2,h3


class encoder(nn.Module):
	def __init__(self, ef_dim):
		super(encoder, self).__init__()
		self.ef_dim = ef_dim
		self.conv_1 = nn.Conv3d(1, self.ef_dim, 4, stride=2, padding=1, bias=True)
		self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim*2, 4, stride=2, padding=1, bias=True)
		self.conv_3 = nn.Conv3d(self.ef_dim*2, self.ef_dim*4, 4, stride=2, padding=1, bias=True)
		self.conv_4 = nn.Conv3d(self.ef_dim*4, self.ef_dim*8, 4, stride=2, padding=1, bias=True)
		self.conv_5 = nn.Conv3d(self.ef_dim*8, self.ef_dim*8, 4, stride=1, padding=0, bias=True)
		nn.init.xavier_uniform_(self.conv_1.weight)
		nn.init.constant_(self.conv_1.bias,0)
		nn.init.xavier_uniform_(self.conv_2.weight)
		nn.init.constant_(self.conv_2.bias,0)
		nn.init.xavier_uniform_(self.conv_3.weight)
		nn.init.constant_(self.conv_3.bias,0)
		nn.init.xavier_uniform_(self.conv_4.weight)
		nn.init.constant_(self.conv_4.bias,0)
		nn.init.xavier_uniform_(self.conv_5.weight)
		nn.init.constant_(self.conv_5.bias,0)

	def forward(self, inputs, is_training=False):
		d_1 = self.conv_1(inputs)
		d_1 = F.leaky_relu(d_1, negative_slope=0.01, inplace=True)

		d_2 = self.conv_2(d_1)
		d_2 = F.leaky_relu(d_2, negative_slope=0.01, inplace=True)

		d_3 = self.conv_3(d_2)
		d_3 = F.leaky_relu(d_3, negative_slope=0.01, inplace=True)

		d_4 = self.conv_4(d_3)
		d_4 = F.leaky_relu(d_4, negative_slope=0.01, inplace=True)

		d_5 = self.conv_5(d_4)
		d_5 = d_5.view(-1, self.ef_dim*8)
		d_5 = torch.sigmoid(d_5)

		return d_5

class decoder(nn.Module):
	def __init__(self, ef_dim, p_dim):
		super(decoder, self).__init__()
		self.ef_dim = ef_dim
		self.p_dim = p_dim
		self.linear_1 = nn.Linear(self.ef_dim*8, self.ef_dim*16, bias=True)
		self.linear_2 = nn.Linear(self.ef_dim*16, self.ef_dim*32, bias=True)
		self.linear_3 = nn.Linear(self.ef_dim*32, self.ef_dim*64, bias=True)
		self.linear_4 = nn.Linear(self.ef_dim*64, self.p_dim*4, bias=True)
		nn.init.xavier_uniform_(self.linear_1.weight)
		nn.init.constant_(self.linear_1.bias,0)
		nn.init.xavier_uniform_(self.linear_2.weight)
		nn.init.constant_(self.linear_2.bias,0)
		nn.init.xavier_uniform_(self.linear_3.weight)
		nn.init.constant_(self.linear_3.bias,0)
		nn.init.xavier_uniform_(self.linear_4.weight)
		nn.init.constant_(self.linear_4.bias,0)

	def forward(self, inputs, is_training=False):
		l1 = self.linear_1(inputs)
		l1 = F.leaky_relu(l1, negative_slope=0.01, inplace=True)

		l2 = self.linear_2(l1)
		l2 = F.leaky_relu(l2, negative_slope=0.01, inplace=True)

		l3 = self.linear_3(l2)
		l3 = F.leaky_relu(l3, negative_slope=0.01, inplace=True)

		l4 = self.linear_4(l3)
		l4 = l4.view(-1, 4, self.p_dim)

		return l4

class bsp_network(nn.Module):
	def __init__(self, phase, ef_dim, p_dim, c_dim):
		super(bsp_network, self).__init__()
		self.phase = phase
		self.ef_dim = ef_dim
		self.p_dim = p_dim
		self.c_dim = c_dim
		self.encoder = encoder(self.ef_dim)
		self.decoder = decoder(self.ef_dim, self.p_dim)
		self.generator = generator(self.phase, self.p_dim, self.c_dim)

	def forward(self, inputs, z_vector, plane_m, point_coord, convex_mask=None, is_training=False):
		if is_training:
			z_vector = self.encoder(inputs, is_training=is_training)
			plane_m = self.decoder(z_vector, is_training=is_training)
			net_out_convexes, net_out = self.generator(point_coord, plane_m, convex_mask=convex_mask, is_training=is_training)
		else:
			if inputs is not None:
				z_vector = self.encoder(inputs, is_training=is_training)
			if z_vector is not None:
				plane_m = self.decoder(z_vector, is_training=is_training)
			if point_coord is not None:
				net_out_convexes, net_out = self.generator(point_coord, plane_m, convex_mask=convex_mask, is_training=is_training)
			else:
				net_out_convexes = None
				net_out = None

		return z_vector, plane_m, net_out_convexes, net_out
