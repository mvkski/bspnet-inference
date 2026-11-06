#!/usr/bin/env python3
"""Host-side Docker launcher. No ML libraries are needed on the host."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from verify_weights import verify


def mount(source, target, readonly=True):
    if any(c in str(source) for c in (",", "\n", "\r")):
        raise ValueError("Docker mount paths cannot contain commas or newlines")
    value = f"type=bind,src={source},dst={target}"
    return ["--mount", value + (",readonly" if readonly else "")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meshes", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--weights-dir", type=Path, default=Path(__file__).resolve().parent / "weights")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--image", default="bspnet-inference:1")
    args = parser.parse_args(argv)
    paths = [p.resolve(strict=True) for p in args.meshes]
    if any(not p.is_file() for p in paths):
        parser.error("Inputs must be files")
    if len({p.stem for p in paths}) != len(paths):
        parser.error("Input mesh stems must be unique")
    if args.gpu < 0:
        parser.error("GPU index must be nonnegative")
    weights = args.weights_dir.resolve(strict=True)
    verify(weights)
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output directory must be new or empty")
    command = ["docker", "run", "--rm", "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}"]
    if args.device == "cuda":
        command += ["--gpus", f"device={args.gpu}"]
    command += mount(weights, "/weights") + mount(output, "/output", readonly=False)
    inputs = []
    for index, path in enumerate(paths):
        command += mount(path.parent, f"/inputs/{index}")
        inputs.append(f"/inputs/{index}/{path.name}")
    command += [args.image, *inputs, "--output", "/output", "--weights-dir", "/weights", "--device", args.device]
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    run = {"status": "running", "inputs": list(map(str, paths)), "weights_dir": str(weights),
           "output_dir": str(output), "command": command}
    run_path = output / "run.json"
    run_path.write_text(json.dumps(run, indent=2) + "\n")
    try:
        with (output / "inference.log").open("w") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        run.update(status="complete" if code == 0 else "failed", returncode=code)
    except BaseException as exc:
        run.update(status="interrupted", error=type(exc).__name__)
        raise
    finally:
        run["command_wall_seconds"] = time.perf_counter() - started
        run_path.write_text(json.dumps(run, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
