"""Deploy dev (D:) -> live (B:) with shutil (robocopy/xcopy silently fail from Git Bash).

    python deploy.py            copy code, tools, tests and docs; the live .venv is left alone
    python deploy.py --dry-run

Not copied: .venv (the live venv is managed separately - see the imagetools-mcp skill's
troubleshooting page for the pinned dependency matrix), benchmarks/, gfpgan/ weights, *.pt,
__pycache__. The live copy must NOT contain an editable install of this package: run_server.py
puts its own folder first on sys.path, and an editable dist in the live venv would resolve
`server` back to D: (2026-07-07 incident).

After deploying: MCPManager `server_reload("image-tools")` (or restart the manager). Running
Claude Code sessions keep their tool list; new tools need a slot or a new session.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
LIVE = Path(r"B:\-AI-Stuff-\-=MCP-Servers=-\ImageTools_MCP")

FILES = ["run_server.py", "pyproject.toml", "README.md", "LICENSE", "deploy.py", "Dockerfile"]
DIRS = ["server", "remote_server", "tools", "tests", "scripts", "docs"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "*.egg-info")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv: list[str]) -> int:
    dry = "--dry-run" in argv
    if not dry:
        LIVE.mkdir(parents=True, exist_ok=True)
    print(f"{SRC} -> {LIVE}{' (dry run)' if dry else ''}")
    for name in FILES:
        src = SRC / name
        if not src.exists():
            continue
        print(f"  {name}")
        if not dry:
            shutil.copy2(src, LIVE / name)
            if _sha(src) != _sha(LIVE / name):
                raise SystemExit(f"hash mismatch after copy: {name}")
    for name in DIRS:
        src = SRC / name
        if not src.is_dir():
            continue
        print(f"  {name}/")
        if not dry:
            dst = LIVE / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=IGNORE)
    if not dry:
        stray = list((LIVE / ".venv" / "Lib" / "site-packages").glob("__editable__*image_tools*")) if (LIVE / ".venv").exists() else []
        if stray:
            print("WARNING: editable install found in the live venv:", [p.name for p in stray])
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
