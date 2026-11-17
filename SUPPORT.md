# ImageTools MCP support

**Reviewed:** November 17, 2026

ImageTools MCP is a local, Python-based image editing and generation server using Model Context Protocol. Its README documents a 242-tool catalog, in-memory canvas and layers, image adjustments, optional segmentation models, and optional GPU-dependent generation features. **A registered tool or README description is not proof that a particular machine has the models, memory or dependencies to execute it.**

## Start with the project guide

- **Install, supported features and examples:** [README](README.md)
- **Public repository and existing issue search:** https://github.com/jabbertones-cloud/image-mcp/issues
- **New reproducible public bug:** https://github.com/jabbertones-cloud/image-mcp/issues/new/choose
- **Upstream MCP protocol documentation:** https://modelcontextprotocol.io/

This repository documents self-hosted software. No public hosted service, ChatGPT directory publication or cloud account authorization is established by these source files.

## Troubleshoot by failure type

| Symptom | What to check and report |
| --- | --- |
| MCP client cannot connect | Client and version, local Python version, environment setup, launch command with secrets removed, transport or tool-discovery error |
| Canvas or layer operation returns an unexpected result | Tool name, redacted minimal input, image dimensions and mode, expected output, actual output, and whether the canvas is still open |
| A segmentation model is missing | Model name, dependency installation, local model availability and download error without access tokens |
| GPU generation fails | GPU/OS, available VRAM, relevant local dependency versions, model selected, and redacted error; optional models are not necessarily installed |
| Image file cannot open or export | Non-sensitive sample image format, dimensions, path shape without private home-directory details, and output format |

Check issues before filing a duplicate. Include a minimal reproducible example rather than an entire private photo collection.

## Privacy and security

Image inputs may contain faces, personal photos, client designs or other sensitive information. Do not attach private images, credentials, raw file-system paths containing personal data, API keys or proprietary model weights to a public GitHub issue. Use the repository owner's private security reporting path when available for vulnerabilities; do not publish exploit details or active credentials.

The maintainer may request further diagnostics through an appropriate private channel, but a public issue should remain non-sensitive. Agents must never create an issue or upload an image without the user's approval and must verify a successful GitHub response before claiming publication.

No GitHub Actions workflows were added or enabled for this documentation update.
