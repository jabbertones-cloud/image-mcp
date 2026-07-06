"""Pre-stage Qwen-Image and Qwen-Image-Edit-2511 weights into the pod's
persistent volume so cold starts don't pay the ~38 GB download every time.

Usage:
  python scripts/download_weights.py --models-root /var/remote-gen/models
                                     --which both         # or qwen-image | qwen-image-edit-2511

Sets HF_HOME under --models-root/hf-cache so subsequent diffusers
``from_pretrained`` calls find the files without going to the network.

Defaults match what ``remote_server.pipeline`` will look for.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPOS = {
    "qwen-image":            os.environ.get("REMOTE_QWEN_IMAGE_REPO", "Qwen/Qwen-Image"),
    "qwen-image-edit-2511":  os.environ.get("REMOTE_QWEN_EDIT_REPO",  "Qwen/Qwen-Image-Edit-2511"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-root", required=True, type=Path)
    ap.add_argument("--which", choices=["both", "qwen-image", "qwen-image-edit-2511"],
                    default="both")
    args = ap.parse_args()

    cache_dir = args.models_root / "hf-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed. pip install huggingface_hub[hf_xet]",
              file=sys.stderr)
        return 2

    targets = list(REPOS.keys()) if args.which == "both" else [args.which]
    for key in targets:
        repo = REPOS[key]
        print(f"[download] {repo} → {cache_dir}")
        # ``transformers_version`` etc. — we let snapshot_download pull
        # the full repo. Skip safetensors mirrors if both exist.
        snapshot_download(
            repo_id=repo,
            cache_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
