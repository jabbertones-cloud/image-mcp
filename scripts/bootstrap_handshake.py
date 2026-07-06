"""Run ONCE before you build the server image.

Generates:
  - A 32-byte PSK, split into two halves
  - A server long-lived identity (X25519 + Ed25519)
  - A client long-lived identity (X25519 + Ed25519)

server-secrets/  → goes into the Docker image / RunPod mount  (do NOT push)
client-secrets/  → goes into ~/.image-tools-remote/ on your PC (do NOT push)

Both directories contain both halves so each side can locally derive the PSK
without ever transmitting either half over the network.
"""
from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path

# project-local import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from remote_server.crypto import (
    Identity, generate_psk, split_psk, write_half,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing bootstrap")
    args = ap.parse_args()

    server_dir = args.out_dir / "server-secrets"
    client_dir = args.out_dir / "client-secrets"
    pairing    = args.out_dir / "pairing.txt"

    for d in (server_dir, client_dir):
        if d.exists() and any(d.iterdir()) and not args.force:
            print(f"refusing to overwrite {d} (use --force to rotate)", file=sys.stderr)
            return 2

    server_dir.mkdir(parents=True, exist_ok=True)
    client_dir.mkdir(parents=True, exist_ok=True)

    # PSK halves
    psk = generate_psk()
    s_half, c_half = split_psk(psk)
    write_half(server_dir / "half.bin",        s_half)
    write_half(server_dir / "client_half.bin", c_half)
    write_half(client_dir / "half.bin",        c_half)
    write_half(client_dir / "server_half.bin", s_half)

    # identities
    server_id = Identity.generate("imagetools-remote-server")
    client_id = Identity.generate(f"imagetools-remote-client-{getpass.getuser()}")

    (server_dir / "identity.json").write_text(
        json.dumps(server_id.to_dict(), indent=2), encoding="utf-8")
    (client_dir / "identity.json").write_text(
        json.dumps(client_id.to_dict(), indent=2), encoding="utf-8")

    # peer pubkey files
    (server_dir / "client_peer.json").write_text(
        json.dumps(client_id.public().to_dict(), indent=2), encoding="utf-8")
    (client_dir / "server_peer.json").write_text(
        json.dumps(server_id.public().to_dict(), indent=2), encoding="utf-8")

    fp_server = server_id.public().fingerprint()
    fp_client = client_id.public().fingerprint()
    pairing.write_text(
        "imagetools-remote-gen pairing\n"
        "==============================\n\n"
        f"server identity fingerprint:\n  {fp_server}\n\n"
        f"client identity fingerprint:\n  {fp_client}\n\n"
        "Bake server-secrets/ into the Docker image at /etc/remote-gen,\n"
        "or pass each file as a REMOTE_*_B64 env var.\n"
        "Copy client-secrets/* to ~/.image-tools-remote/ on your PC.\n",
        encoding="utf-8",
    )

    print("\n=== bootstrap complete ===")
    print(f"  server secrets: {server_dir.resolve()}")
    print(f"  client secrets: {client_dir.resolve()}")
    print(f"  server fingerprint: {fp_server}")
    print(f"  client fingerprint: {fp_client}")
    print(f"  pairing summary:   {pairing.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
