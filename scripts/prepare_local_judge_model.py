"""Check the Qwen3-8B Hugging Face cache; optionally download missing files."""
import argparse
import json
from pathlib import Path


MODEL = "Qwen/Qwen3-8B"


def complete(snapshot):
    root = Path(snapshot)
    required = [root / "config.json", root / "tokenizer.json", root / "tokenizer_config.json"]
    index = root / "model.safetensors.index.json"
    if index.is_file():
        try:
            shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        except (ValueError, KeyError, TypeError):
            return False
        required.extend(root / name for name in shards)
    else:
        required.append(root / "model.safetensors")
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def resolve(model=MODEL, revision="main", download=False):
    path = Path(model).expanduser()
    if path.exists():
        if not complete(path):
            raise RuntimeError(f"Incomplete local model directory: {path}")
        return path.resolve(), "local-path"
    from huggingface_hub import snapshot_download
    try:
        snapshot = snapshot_download(repo_id=model, revision=revision, local_files_only=True)
    except (OSError, ValueError):
        snapshot = None
    if snapshot and complete(snapshot):
        return Path(snapshot).resolve(), "cached"
    if not download:
        raise RuntimeError(f"{model} is absent/incomplete in this Hugging Face cache")
    snapshot = snapshot_download(repo_id=model, revision=revision)
    if not complete(snapshot):
        raise RuntimeError(f"Download incomplete: {snapshot}")
    return Path(snapshot).resolve(), "downloaded"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL, help="HF repo ID or complete local snapshot path")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--download-if-missing", action="store_true")
    args = parser.parse_args()
    try:
        path, state = resolve(args.model, args.revision, args.download_if_missing)
    except RuntimeError as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(dict(state=state, model=args.model, snapshot=str(path),
                          revision=path.name if len(path.name) == 40 else args.revision), indent=2))


if __name__ == "__main__":
    main()
