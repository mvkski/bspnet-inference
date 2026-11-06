"""Verify the inference-tested checkpoint without importing PyTorch."""
import argparse
import hashlib
from pathlib import Path

FILENAME = "BSP_AE_final.pth"
EXPECTED_SHA256 = "50da9886e340e742de9522171004ac00aa2288cc80c9f37b93d852e6b4b1985e"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(directory):
    path = Path(directory) / FILENAME
    if sha256(path) != EXPECTED_SHA256:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(f"Verified: {verify(args.directory)}")
