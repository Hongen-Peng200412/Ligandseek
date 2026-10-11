"""Download public release weights and selected examples with SHA-256 verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("unet_raw", "unet_diff", "unet_full", "ligandseek_pv")
CASES = ("8z0f", "7z24", "9bxc")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def extract_example(archive_path: Path, assets_root: Path, case: str) -> None:
    """Extract only regular files/directories below examples/<case>, with bounded expansion."""
    root = assets_root.resolve()
    staging = Path(tempfile.mkdtemp(prefix=".unpack_", dir=root))
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            if len(members) > 1024 or sum(member.size for member in members) > 4 * 1024**3:
                raise ValueError("Example archive exceeds the member or expanded-size limit.")
            seen = set()
            for member in members:
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or ".." in relative.parts or "\\" in member.name or ":" in member.name:
                    raise ValueError(f"Unsafe archive path: {member.name}")
                if relative.parts == ("examples",) and member.isdir():
                    continue
                if relative.parts[:2] != ("examples", case) or not (member.isdir() or member.isfile()):
                    raise ValueError(f"Unsupported archive member: {member.name}")
                if relative in seen:
                    raise ValueError(f"Duplicate archive member: {member.name}")
                seen.add(relative)
                target = staging.joinpath(*relative.parts)
                target.resolve().relative_to(staging.resolve())
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        for path in sorted(staging.rglob("*")):
            relative = path.relative_to(staging)
            destination = root / relative
            destination.resolve().relative_to(root)
            if path.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, destination)
    finally:
        staging.resolve().relative_to(root)
        shutil.rmtree(staging)


# ================================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=(*CASES, "all"), default="8z0f")
    parser.add_argument("--model", choices=(*MODELS, "all"), default="all")
    parser.add_argument("--assets-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--manifest", type=Path, default=ROOT / "assets_manifest.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    root = args.assets_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected = [asset for asset in manifest["assets"] if (asset["kind"] == "checkpoint" and (args.model == "all" or asset["model"] == args.model)) or (asset["kind"] == "example" and (args.example == "all" or asset["case"] == args.example))]
    expected_models = MODELS if args.model == "all" else (args.model,)
    expected_cases = CASES if args.example == "all" else (args.example,)
    if {asset["model"] for asset in selected if asset["kind"] == "checkpoint"} != set(expected_models) or {asset["case"] for asset in selected if asset["kind"] == "example"} != set(expected_cases):
        raise ValueError("Release manifest does not contain every selected weight and example.")
    for asset in selected:
        relative = PurePosixPath(asset["destination"])
        if relative.is_absolute() or ".." in relative.parts or ":" in asset["destination"] or "\\" in asset["destination"]:
            raise ValueError("Unsafe asset destination in manifest.")
        if asset["kind"] == "example":
            if relative.parts:
                raise ValueError("Example archives must extract at the assets root.")
            if Path(asset["name"]).name != asset["name"]:
                raise ValueError("Archive name must be a basename.")
            destination = root / ".downloads" / asset["name"]
        else:
            if relative.parts[:1] != ("checkpoints",):
                raise ValueError("Checkpoint destinations must be below checkpoints/.")
            destination = root.joinpath(*relative.parts)
        destination.resolve().relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected_size = int(asset["bytes"])
        cached = destination.is_file() and destination.stat().st_size == expected_size and file_sha256(destination) == asset["sha256"]
        if not cached:
            if not asset["url"].startswith("https://"):
                raise ValueError("Release downloads require HTTPS.")
            temporary = destination.with_name(destination.name + ".part")
            print(f'Downloading {asset["name"]} ({expected_size / 1024**2:.1f} MiB)', flush=True)
            try:
                request = urllib.request.Request(asset["url"], headers={"User-Agent": "Ligandseek/1.0", "Accept": "application/octet-stream"})
                with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                    downloaded = 0
                    while chunk := response.read(8 * 1024 * 1024):
                        downloaded += len(chunk)
                        if downloaded > expected_size:
                            raise ValueError("Download is larger than its release manifest entry.")
                        output.write(chunk)
                if temporary.stat().st_size != expected_size or file_sha256(temporary) != asset["sha256"]:
                    raise ValueError(f'Checksum or size mismatch for {asset["name"]}.')
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
        print(f'Verified {asset["name"]}', flush=True)
        if asset["kind"] == "example":
            extract_example(destination, root, asset["case"])
            print(f'Prepared examples/{asset["case"]}', flush=True)
    print(f"Assets ready: {root}", flush=True)


if __name__ == "__main__":
    main()
