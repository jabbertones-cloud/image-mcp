# Remote generation — server returns latents, client decodes locally

`remote_server/` is a FastAPI app deployed to a GPU pod (RunPod H100 by
default) that runs **one** diffusers pipeline per pod. The server runs
the expensive DiT denoising and returns **a sealed latent tensor** — not
a decoded image. The local MCP server (your laptop / workstation) runs
the matching VAE locally to decode the latent into a PIL image.

This page explains the why, the wire format, and the deploy steps.

## Why latents over the wire, not images

| Concern | Latents | PNG |
|---|---|---|
| Size (1024×1024 Qwen-Image) | ~128 KB (1×16×64×64 bf16) | ~3–5 MB |
| Server VRAM | DiT + text encoder only (~38 GB) | + VAE (~250 MB more) |
| Server sees the image? | No — just opaque floats | Yes |
| Re-decode with different VAE / tiling? | Free | Re-pay denoising |
| Resilience to GPU model swap? | Server-side decoder model swap doesn't break clients | Tied to server's exact VAE version |

The client's local GPU does the VAE pass in ~0.5 s on a 4090.

## Architecture

```
┌──── client (local) ──────────────┐         ┌──── server (cloud GPU) ─────┐
│ MCP tool: remote_qwen_txt2img   │         │ FastAPI app (port 8443)     │
│  └─ sign manifest, seal payload │         │  └─ verify proof + signature │
│  └─ POST /v1/generate/txt2img ──┼─encrypt─┼─→ pipeline.generate(...)    │
│                                  │         │     output_type='latent'   │
│  ←─sealed safetensors blob──────┼─encrypt─┼─ pack_latent_blob(...)      │
│  └─ unpack_latent_blob          │         │  └─ seal to client pubkey   │
│  └─ local VAE decode → PIL.Image│         │     return                  │
│  └─ paste onto MCP canvas       │         └──────────────────────────────┘
└──────────────────────────────────┘
```

Crypto + auth pattern is identical to QwenCharLoRA's split-PSK +
signed-manifest + sealed-blob design. The `remote_server/crypto.py` and
`remote_server/secure_store.py` files are byte-synced copies — see the
sync-note headers.

## Models supported

| `REMOTE_MODEL` | Class | Endpoint | Mode |
|---|---|---|---|
| `qwen-image` | `QwenImagePipeline` | `POST /v1/generate/txt2img` | text → latent |
| `qwen-image-edit-2511` | `QwenImageEditPipeline` | `POST /v1/generate/img2img` | text + control image → latent |

**One model per pod.** No swapping at runtime — cold-start memory
budgeting is cleaner, and the auto-routing logic on the client side is
trivial (pod fingerprint carries the model name; client refuses
mismatch).

## Wire format

### Request envelope

Symmetric with QwenCharLoRA. Body = `WrappedBlob` (X25519 SealedBox)
wrapping JSON:

```json
{
  "manifest_signed_hex": "<128-char hex of Ed25519-signed manifest>",
  "payload": {
    "_kind":             "txt2img" | "img2img",
    "prompt":            "...",
    "negative_prompt":   "",
    "width":             1024,
    "height":            1024,
    "steps":             20,
    "cfg":               4.0,
    "seed":              12345,
    "control_image_b64": "<base64 PNG (img2img only)>"
  }
}
```

`manifest` is the signed object; it carries
`payload_sha256 = sha256(canonical-json(payload))` which the server
recomputes from the supplied payload to prevent envelope substitution.

Headers (every call):

```
X-RGEN-Request-Id: <uuid4 hex>
X-RGEN-Proof:      <urlsafe-base64 HMAC-SHA256 keyed by combined PSK
                    over "qcl-proof-v1|<request_id>|<sha256(body)>" >
```

### Response

`Content-Type: application/octet-stream` — the bytes are:

```
WrappedBlob(client.enc_pub) {
    safetensors blob {
        "latent":  Tensor (B, 16, H/16, W/16) bf16,
        __metadata__ = {
            "__json__": json.dumps({
                model, seed, steps, cfg, height, width,
                scaling, vae_required
            })
        }
    }
}
```

The client unwraps with its X25519 priv, parses safetensors, looks at
`vae_required`, lazy-loads the matching local VAE, decodes, returns a
PIL image.

## File layout enforced by `settings.py`

| Path | Persistence | Holds |
|---|---|---|
| `/etc/remote-gen/` | persistent | server identity + PSK halves (server-side secrets only — no client data) |
| `/var/remote-gen/models/` | persistent volume | public Qwen-Image / Qwen-Image-Edit weights — no PII |
| `/run/remote-gen/transfer/` | **tmpfs (enforced)** | every byte of client data-in-flight; cleared per request |

The FastAPI lifespan calls `assert_transfer_is_tmpfs()` and refuses to
start if the path is not a tmpfs mountpoint. Override for local dev
with `REMOTE_TRANSFER_DIR_SKIP_TMPFS_CHECK=1`.

## Deploy (RunPod)

1. `python scripts/bootstrap_handshake.py --out-dir ./build`
2. (optional) `python -c "from remote_server.secure_store import SecretFile; SecretFile('build/client-secrets/half.bin.enc').write(open('build/client-secrets/half.bin','rb').read(), 'YourPassphrase')"` then shred the plaintext.
3. `docker build -t imagetools-remotegen:0.1 .`
4. Push to a registry, create RunPod template with:
   - GPU: H100 PCIe 80 GB
   - Exposed port: 8443 (or skip if using Tailscale)
   - Env vars: `REMOTE_*_B64` (base64 of each `build/server-secrets/` file), `REMOTE_MODEL=qwen-image` (or `qwen-image-edit-2511`), `TS_AUTHKEY=tskey-auth-…` (optional)
   - Volume mounts:
     - `/var/remote-gen/models` ← persistent volume with Qwen weights
     - `/run/remote-gen/transfer` ← tmpfs, size 4G, mode 1700
5. Start. The entrypoint prints the server identity fingerprint to the
   logs — verify against `build/pairing.txt` before sending any data.
6. On the client (your PC):
   ```powershell
   New-Item -ItemType Directory $env:USERPROFILE\.image-tools-remote
   Copy-Item build\client-secrets\* $env:USERPROFILE\.image-tools-remote\
   @'
   {"server_url": "https://<pod>-8443.proxy.runpod.net",
    "server_fingerprint": "<paste from pairing.txt>"}
   '@ | Set-Content $env:USERPROFILE\.image-tools-remote\config.json
   ```
7. From the MCP client (Claude / your tool): call
   `remote_status` first to verify the fingerprint, then
   `remote_qwen_txt2img(prompt=...)`. The image lands on a fresh canvas.

## MCP tools

| Tool | What it does |
|---|---|
| `remote_status()` | GETs `/v1/pubkey`, verifies signature, returns `{label, fingerprint, model}` — call this first. |
| `remote_qwen_txt2img(prompt, …, canvas_id?)` | Generates with `qwen-image`, decodes locally, places result on a new (or specified) canvas. |
| `remote_qwen_edit(canvas_id, prompt, …)` | Sends the active layer of `canvas_id` to a `qwen-image-edit-2511` pod, decodes, places result as a new layer on the same canvas. |

## What's NOT in this prototype

- LoRA injection on the server (planned: include LoRA b64 bytes in the
  payload, server hot-loads them per request, server-side LRU cache).
- ControlNet variants (Canny / Depth / OpenPose).
- Multi-model resident pods.
- Live preview latents during denoising (every N steps).

## Local development without GPU

The test suite uses a stub pipeline that returns deterministic zero
tensors — exercises the full crypto chain + route wiring + sealed-latent
return without diffusers or a CUDA device. Set
`REMOTE_SKIP_PIPELINE_LOAD=1` to start the server without loading the
real pipeline.
