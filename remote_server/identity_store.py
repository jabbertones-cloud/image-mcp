"""Server-side identity + half-key loading.

Files expected under ``secrets_dir``:
  half.bin             — 32-byte server half of the split PSK
  client_half.bin      — 32-byte client half (server has both; combines at boot)
  identity.json        — server's X25519 + Ed25519 keypair
  client_peer.json     — paired client's public-only material
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from remote_server.crypto import Identity, PublicIdentity, read_half

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServerIdentityStore:
    self_identity: Identity
    paired_client: PublicIdentity
    server_half: bytes
    client_half: bytes


def load_store(secrets_dir: Path) -> ServerIdentityStore:
    secrets_dir = Path(secrets_dir)
    half_path        = secrets_dir / "half.bin"
    client_half_path = secrets_dir / "client_half.bin"
    identity_path    = secrets_dir / "identity.json"
    peer_path        = secrets_dir / "client_peer.json"

    for p in (half_path, client_half_path, identity_path, peer_path):
        if not p.exists():
            raise FileNotFoundError(
                f"missing required secret {p}\n"
                "Bake the bootstrap server-secrets/ into the image at "
                "/etc/remote-gen, or set REMOTE_SECRETS_DIR.")

    server_half = read_half(half_path)
    client_half = read_half(client_half_path)
    self_id = Identity.from_dict(json.loads(identity_path.read_text(encoding="utf-8")))
    peer    = PublicIdentity.from_dict(json.loads(peer_path.read_text(encoding="utf-8")))

    log.info(f"identity: self={self_id.label}  peer={peer.label}")
    log.info(f"  server fingerprint: {self_id.public().fingerprint()}")
    log.info(f"  client fingerprint: {peer.fingerprint()}")
    return ServerIdentityStore(
        self_identity=self_id,
        paired_client=peer,
        server_half=server_half,
        client_half=client_half,
    )
