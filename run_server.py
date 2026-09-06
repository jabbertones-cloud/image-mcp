"""Launcher for claude-code registration.

Claude Code spawns MCP servers without a working directory, so module imports
relative to the project root fail. This shim adds its own directory to
``sys.path`` before importing the server entry point.

The heavy AI imports (torch / diffusers / transformers / scipy) are pre-warmed
in a **background thread** so the FastMCP server responds to MCP ``initialize``
immediately. The original blocking approach timed out the MCP client (~30 s
connection timeout vs ~60 s import cost). The daemon thread completes in the
background; by the time the first heavy tool call arrives (e.g. ``qwen_load``),
the imports are cached.

NOTE: the prewarm MUST run on the main thread for the ``scipy.linalg.blas``
DLL-deadlock workaround to work (see prior troubleshooting notes). However,
the MCP client timeout forces our hand. As a compromise, we import the
LIGHTWEIGHT modules (torch, transformers, diffusers top-level) on the main
thread (~5 s), then defer the heavy sub-imports (pipeline classes that pull in
scipy) to the background thread.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# transformers v5's parallel weight materializer access-violates on Windows;
# fall back to the single-threaded loader.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")


def _prewarm_light() -> None:
    """Quick main-thread imports of the top-level modules (~5 s warm). They prime
    torch's DLLs so the background thread never fights the main thread over them."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import diffusers  # noqa: F401
    except Exception:
        pass


# Phase 1 - main thread: top-level modules.
_prewarm_light()

# Phase 2 - every pipeline / model class the AI tools use, imported ONCE on the
# main thread (server/prewarm.py) before FastMCP starts. Two lessons from
# 2026-09-07 (py-spy proof in that module): concurrent lazy imports from two
# threads deadlock the event loop, and scipy's native extensions freeze when
# first imported from a non-main thread on Windows. MCPManager's
# startup_timeout_s for this server covers the ~1-2 min cold start; the child
# stays resident afterwards. Do not move this back to a thread on Windows.
from server import prewarm  # noqa: E402

prewarm.start()

from server.image_tools_server import main

if __name__ == "__main__":
    main()
