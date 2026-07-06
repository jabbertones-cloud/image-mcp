"""ImageTools remote generation server.

This subpackage is deployed to a GPU pod (RunPod H100 by default).
It loads ONE diffusers pipeline per pod (selected via REMOTE_MODEL env var),
accepts encrypted generation requests, and returns SEALED LATENTS that the
client decodes locally with its own VAE. The server never decodes to RGB.

See ``D:\\App Dev\\ImageTools_MCP\\docs\\REMOTE_GEN.md`` for the protocol
and ``Dockerfile`` for the deploy stack (mirrors the QwenCharLoRA pattern).
"""
__version__ = "0.1.0"
