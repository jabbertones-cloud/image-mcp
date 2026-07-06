#!/usr/bin/env bash
# ImageTools remote-gen server entrypoint.

set -euo pipefail

SECRETS_DIR="${REMOTE_SECRETS_DIR:-/etc/remote-gen}"
MODELS_ROOT="${REMOTE_MODELS_ROOT:-/var/remote-gen/models}"
TRANSFER_DIR="${REMOTE_TRANSFER_DIR:-/run/remote-gen/transfer}"
PORT="${REMOTE_PORT:-8443}"
HOST="${REMOTE_HOST:-0.0.0.0}"
MODEL="${REMOTE_MODEL:-qwen-image}"

TS_STATE_DIR="${TS_STATE_DIR:-/var/lib/tailscale}"
TS_SOCKET="${TS_SOCKET:-/var/run/tailscale/tailscaled.sock}"
TS_PID=""

log() { echo "[remote-gen-entrypoint] $*"; }

# ── 1. secrets ──────────────────────────────────────────────────────

mkdir -p "$SECRETS_DIR"
inject_b64() {
    local var="$1" file="$2"
    local val="${!var:-}"
    if [[ -n "$val" && ! -f "$SECRETS_DIR/$file" ]]; then
        log "injecting $file from env $var"
        echo "$val" | base64 -d > "$SECRETS_DIR/$file"
        chmod 600 "$SECRETS_DIR/$file"
    fi
}
inject_b64 REMOTE_SERVER_HALF_B64     half.bin
inject_b64 REMOTE_CLIENT_HALF_B64     client_half.bin
inject_b64 REMOTE_SERVER_IDENTITY_B64 identity.json
inject_b64 REMOTE_CLIENT_PEER_B64     client_peer.json

required=(half.bin client_half.bin identity.json client_peer.json)
missing=()
for f in "${required[@]}"; do
    [[ -f "$SECRETS_DIR/$f" ]] || missing+=("$f")
done
if (( ${#missing[@]} > 0 )); then
    log "FATAL: missing secrets in $SECRETS_DIR: ${missing[*]}"
    exit 1
fi
chmod -R go-rwx "$SECRETS_DIR" || true

# ── 2. transfer dir MUST be tmpfs (the "every transfer file in RAM" guarantee) ──

mkdir -p "$TRANSFER_DIR"
if ! mountpoint -q "$TRANSFER_DIR"; then
    log "FATAL: $TRANSFER_DIR is not a tmpfs mountpoint."
    log "       Every byte of client data-in-flight must live in RAM."
    log "       Mount it with:"
    log "         docker run ... --tmpfs $TRANSFER_DIR:size=4g,mode=1700 ..."
    log "       or on RunPod: template -> volume_mounts -> type=tmpfs,target=$TRANSFER_DIR"
    log "       To bypass for local dev set REMOTE_TRANSFER_DIR_SKIP_TMPFS_CHECK=1"
    if [[ "${REMOTE_TRANSFER_DIR_SKIP_TMPFS_CHECK:-0}" != "1" ]]; then
        exit 1
    fi
    log "       WARN: bypass active; transfer files may end up on the host disk"
fi
chmod 700 "$TRANSFER_DIR" || true

# ── 3. weights present? check the HF cache — pipeline downloads on demand,
#       but pre-staging keeps cold-starts deterministic and avoids the network
#       fetch happening under the first user request. ─────────────────

case "$MODEL" in
    qwen-image)
        repo_dir="models--Qwen--Qwen-Image" ;;
    qwen-image-edit-2511)
        repo_dir="models--Qwen--Qwen-Image-Edit-2511" ;;
    *)
        log "FATAL: unsupported REMOTE_MODEL=$MODEL"; exit 1 ;;
esac
hf_cache="$MODELS_ROOT/hf-cache"
if [[ ! -d "$hf_cache/$repo_dir" ]]; then
    log "WARN: HF cache at $hf_cache/$repo_dir missing for model $MODEL"
    log "      The pipeline will attempt to download on first request — slow."
    log "      Pre-stage via:"
    log "        python /opt/remote-server/scripts/download_weights.py \\\\"
    log "            --models-root $MODELS_ROOT --which $MODEL"
fi
export HF_HOME="$hf_cache"
export HF_HUB_CACHE="$hf_cache"

# ── 4. tailscale (optional, gated on TS_AUTHKEY) ────────────────────

start_tailscale() {
    [[ -z "${TS_AUTHKEY:-}" ]] && { log "TS_AUTHKEY not set — skipping Tailscale"; return 0; }
    log "starting tailscaled (userspace networking)"
    mkdir -p "$TS_STATE_DIR" "$(dirname "$TS_SOCKET")"
    /usr/sbin/tailscaled \
        --tun=userspace-networking \
        --state="$TS_STATE_DIR/tailscaled.state" \
        --socket="$TS_SOCKET" \
        > /var/log/tailscaled.log 2>&1 &
    TS_PID=$!
    for _ in $(seq 1 30); do
        [[ -S "$TS_SOCKET" ]] && break
        sleep 1
    done
    if [[ ! -S "$TS_SOCKET" ]]; then
        log "ERROR: tailscaled socket never appeared"
        tail -n 20 /var/log/tailscaled.log >&2 || true
        return 1
    fi
    local hostname="${TS_HOSTNAME:-remote-gen-$(cat /etc/hostname 2>/dev/null || echo unknown)}"
    /usr/bin/tailscale --socket="$TS_SOCKET" up \
        --authkey="$TS_AUTHKEY" \
        --hostname="$hostname" \
        --accept-routes \
        --reset \
        ${TS_EXTRA_ARGS:-} || return 1
    /usr/bin/tailscale --socket="$TS_SOCKET" serve --bg \
        --https=443 / "http://127.0.0.1:${PORT}" || \
        log "WARN: tailscale serve failed"
    local ts_url
    ts_url=$(/usr/bin/tailscale --socket="$TS_SOCKET" status --json 2>/dev/null \
        | jq -r '.Self.DNSName // empty' | sed 's/\.$//')
    if [[ -n "$ts_url" ]]; then
        log "  tailscale URL: https://${ts_url}/"
        log "  short URL    : https://$(echo "$ts_url" | cut -d. -f1)/"
    fi
}

stop_tailscale() {
    [[ -z "$TS_PID" ]] && return
    /usr/bin/tailscale --socket="$TS_SOCKET" logout 2>/dev/null || true
    kill "$TS_PID" 2>/dev/null || true
    wait "$TS_PID" 2>/dev/null || true
}

trap stop_tailscale EXIT
start_tailscale || log "WARN: tailscale init failed; continuing without"

# ── 5. report identity ──────────────────────────────────────────────

python - <<'PY' || true
import os, json, sys
sys.path.insert(0, '/opt/remote-server')
from remote_server.crypto import Identity
from pathlib import Path
d = Path(os.environ.get("REMOTE_SECRETS_DIR","/etc/remote-gen"))
ident = Identity.from_dict(json.loads((d / "identity.json").read_text()))
print(f"[remote-gen-entrypoint] server identity fingerprint: {ident.public().fingerprint()}")
print(f"[remote-gen-entrypoint] loaded model:               {os.environ.get('REMOTE_MODEL','qwen-image')}")
PY

# ── 6. uvicorn ──────────────────────────────────────────────────────

log "starting uvicorn on $HOST:$PORT"
exec python -m uvicorn remote_server.app:app \
    --host "$HOST" --port "$PORT" \
    --log-level info --no-access-log
