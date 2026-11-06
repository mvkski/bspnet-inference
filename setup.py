from setuptools import Extension, setup
from Cython.Build import cythonize

setup(name="bspnet-hy3d-inference", version="1.0.0", packages=["bspnet"],
      ext_modules=cythonize([Extension("bspnet.bspt", ["bspnet/bspt.pyx"])],
                            compiler_directives={"language_level": 3}))
