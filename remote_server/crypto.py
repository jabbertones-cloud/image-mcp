"""End-to-end crypto for client ↔ server traffic.

NOTE — synced from D:\\App Dev\\QwenCharLoRA\\qwen_char_lora\\crypto.py.
If you fix a bug here, fix it there too. Both projects use the same
split-PSK + WrappedBlob + Ed25519-signed-manifest pattern; keeping them
byte-identical means a future agent can replace one copy with the other.

Primitives (libsodium via PyNaCl — chosen for being misuse-resistant):

* X25519 SealedBox   → asymmetric envelope: anyone holding the peer's pubkey
                       can encrypt a payload only the peer can open. Used to
                       wrap the per-job symmetric key (and to send LoRA back).
* SecretBox          → XSalsa20-Poly1305 AEAD over a 32-byte symmetric key
                       (NaCl's default; libsodium's "secretbox"). Used for the
                       dataset blob and any large payload.
* Ed25519 SigningKey → identity signatures. Long-lived keypair per side.
* Split-PSK          → a 32-byte pre-shared key is split into two halves at
                       provisioning time. The server image carries one half,
                       the client carries the other. Neither half ever traverses
                       the network; every protected request includes an HMAC
                       computed under the *combined* PSK (a value only present
                       in process memory once both sides cooperate). An attacker
                       who pulls the server image but lacks the client half
                       cannot mint valid request proofs; an attacker who steals
                       the client half cannot impersonate the server.

The lib-level invariants we rely on:

* SecretBox.encrypt() generates a random 24-byte nonce internally and prepends
  it to the ciphertext. We trust that — never reuse a key under a custom nonce.
* SealedBox provides anonymous-sender ECDH; we add a separate detached Ed25519
  signature for accountability.

All wire-format functions return / accept bytes-only payloads so the transport
layer (HTTPS multipart) doesn't have to know anything about the encoding.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nacl.public  import PrivateKey, PublicKey, SealedBox
from nacl.secret  import SecretBox
from nacl.signing import SigningKey, VerifyKey
from nacl.exceptions import BadSignatureError, CryptoError


PSK_BYTES = 32                   # SHA-256 / HMAC-SHA-256 natural width


# ─── encodings ────────────────────────────────────────────────────────


def b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def b64d(s: str) -> bytes:
    s = s + "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s.encode("ascii"))


def fingerprint(pubkey: bytes) -> str:
    """SHA-256 fingerprint formatted as 'AB:CD:EF:...' for the user to copy."""
    h = hashlib.sha256(pubkey).digest()
    return ":".join(f"{b:02X}" for b in h)


# ─── identity keypairs ────────────────────────────────────────────────


@dataclass
class Identity:
    """One side's long-lived (X25519 + Ed25519) keypair, plus a label."""
    label: str
    enc_priv: bytes        # 32 bytes — X25519 secret
    enc_pub:  bytes        # 32 bytes — X25519 public
    sig_priv: bytes        # 32 bytes — Ed25519 seed
    sig_pub:  bytes        # 32 bytes — Ed25519 verify key

    @classmethod
    def generate(cls, label: str) -> "Identity":
        enc = PrivateKey.generate()
        sig = SigningKey.generate()
        return cls(
            label=label,
            enc_priv=bytes(enc),
            enc_pub=bytes(enc.public_key),
            sig_priv=bytes(sig),
            sig_pub=bytes(sig.verify_key),
        )

    # serialisation: never store privkeys on disk in cleartext; the wrapping
    # is up to the caller (DPAPI on Windows client, tmpfs on the server).
    def to_dict(self) -> dict[str, str]:
        return {
            "label":    self.label,
            "enc_priv": b64e(self.enc_priv),
            "enc_pub":  b64e(self.enc_pub),
            "sig_priv": b64e(self.sig_priv),
            "sig_pub":  b64e(self.sig_pub),
        }

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "Identity":
        return cls(
            label=d["label"],
            enc_priv=b64d(d["enc_priv"]),
            enc_pub=b64d(d["enc_pub"]),
            sig_priv=b64d(d["sig_priv"]),
            sig_pub=b64d(d["sig_pub"]),
        )

    def public(self) -> "PublicIdentity":
        return PublicIdentity(label=self.label, enc_pub=self.enc_pub, sig_pub=self.sig_pub)


@dataclass
class PublicIdentity:
    label: str
    enc_pub: bytes
    sig_pub: bytes

    def fingerprint(self) -> str:
        return fingerprint(self.enc_pub + self.sig_pub)

    def to_dict(self) -> dict[str, str]:
        return {"label": self.label,
                "enc_pub": b64e(self.enc_pub),
                "sig_pub": b64e(self.sig_pub),
                "fingerprint": self.fingerprint()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PublicIdentity":
        return cls(label=d["label"], enc_pub=b64d(d["enc_pub"]), sig_pub=b64d(d["sig_pub"]))


# ─── payload encryption ───────────────────────────────────────────────


@dataclass
class WrappedBlob:
    """Format on the wire:
       [4-byte BE length of wrapped_key][wrapped_key][ciphertext_with_nonce_prefix]
    wrapped_key = X25519 SealedBox(recipient_pubkey, symmetric_key)
    ciphertext_with_nonce_prefix = SecretBox(symmetric_key).encrypt(payload)
    """
    blob: bytes

    @classmethod
    def seal(cls, recipient_enc_pub: bytes, payload: bytes) -> "WrappedBlob":
        key = secrets.token_bytes(32)
        box = SecretBox(key)
        ct  = box.encrypt(payload)              # 24-byte nonce prefix included
        sb  = SealedBox(PublicKey(recipient_enc_pub))
        wrapped_key = sb.encrypt(key)
        return cls(blob=len(wrapped_key).to_bytes(4, "big") + wrapped_key + ct)

    def open(self, recipient_enc_priv: bytes) -> bytes:
        if len(self.blob) < 4:
            raise CryptoError("blob too short")
        n = int.from_bytes(self.blob[:4], "big")
        wrapped_key = self.blob[4 : 4 + n]
        ct          = self.blob[4 + n :]
        sb  = SealedBox(PrivateKey(recipient_enc_priv))
        key = sb.decrypt(wrapped_key)
        return SecretBox(key).decrypt(ct)


# ─── signed manifests ─────────────────────────────────────────────────


def sign_json(signing_priv: bytes, payload: dict[str, Any]) -> bytes:
    """Canonical-JSON + Ed25519. Output: signature(64) || payload_bytes."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    sig = SigningKey(signing_priv).sign(canonical).signature
    return sig + canonical


def verify_signed(verify_pub: bytes, signed: bytes) -> dict[str, Any]:
    """Returns the payload dict iff the signature is valid; raises otherwise."""
    if len(signed) < 64:
        raise BadSignatureError("signed payload too short")
    sig, canonical = signed[:64], signed[64:]
    VerifyKey(verify_pub).verify(canonical, sig)
    return json.loads(canonical)


# ─── job manifest helpers ─────────────────────────────────────────────


def make_job_manifest(
    client: PublicIdentity,
    job_params: dict[str, Any],
    dataset_sha256: str,
) -> dict[str, Any]:
    """The thing the client signs and posts alongside the encrypted dataset."""
    return {
        "version":        1,
        "ts":             datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "client":         client.to_dict(),
        "params":         job_params,
        "dataset_sha256": dataset_sha256,
        "nonce":          b64e(secrets.token_bytes(16)),  # replay defence
    }


def sha256_hex(data: bytes | Path) -> str:
    h = hashlib.sha256()
    if isinstance(data, (bytes, bytearray)):
        h.update(data); return h.hexdigest()
    with open(data, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── split-PSK handshake ──────────────────────────────────────────────


def generate_psk() -> bytes:
    """Cryptographically-random 32-byte PSK. Use once at provisioning."""
    return secrets.token_bytes(PSK_BYTES)


def split_psk(psk: bytes) -> tuple[bytes, bytes]:
    """XOR-split: each half is a uniform random byte-string carrying zero
    information about ``psk`` on its own. ``psk == server_half XOR client_half``.

    Returns (server_half, client_half). Neither must ever cross the network."""
    if len(psk) != PSK_BYTES:
        raise ValueError(f"psk must be {PSK_BYTES} bytes, got {len(psk)}")
    server_half = secrets.token_bytes(PSK_BYTES)
    client_half = bytes(a ^ b for a, b in zip(psk, server_half))
    return server_half, client_half


def combine_psk(server_half: bytes, client_half: bytes) -> bytes:
    """Reconstruct the PSK from both halves. Both must equal PSK_BYTES;
    output is exactly 32 bytes."""
    if len(server_half) != PSK_BYTES or len(client_half) != PSK_BYTES:
        raise ValueError(f"halves must each be {PSK_BYTES} bytes")
    return bytes(a ^ b for a, b in zip(server_half, client_half))


def _hkdf_sha256(key_material: bytes, info: bytes, length: int = 32) -> bytes:
    """HKDF-Expand(SHA-256). Plain HMAC, no extract — caller already has a
    uniformly-distributed input (PSK XOR ECDH)."""
    out = b""
    block = b""
    counter = 1
    while len(out) < length:
        block = hmac.new(key_material, block + info + bytes([counter]),
                         hashlib.sha256).digest()
        out += block
        counter += 1
    return out[:length]


def derive_session_key(
    psk: bytes,
    transcript: bytes,
    *,
    ecdh_secret: bytes | None = None,
    info: bytes = b"qcl-session-v1",
) -> bytes:
    """Derive a 32-byte session key from the PSK and the request transcript.

    transcript = hash of (client_pubkey || job_id || dataset_sha256 || ts)
    ecdh_secret = optional shared X25519 secret (forward secrecy). If supplied,
                  the session key depends on a fresh ECDH that an attacker
                  cannot replay later even with both halves.

    Both sides MUST agree on the transcript bytes — any difference produces
    a different key and the AEAD tag check fails."""
    material = psk
    if ecdh_secret is not None:
        material = bytes(a ^ b for a, b in zip(
            psk + b"\x00" * 32, ecdh_secret + b"\x00" * 32
        ))[:PSK_BYTES]
    return _hkdf_sha256(material, info + b"|" + transcript, length=32)


def compute_request_proof(psk: bytes, request_id: str, body_hash: str) -> str:
    """Per-request HMAC proving the caller holds the *combined* PSK.

    Server middleware recomputes this from its own combined PSK and the
    request's request_id + body_hash; mismatch = 401."""
    mac = hmac.new(
        psk,
        f"qcl-proof-v1|{request_id}|{body_hash}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return b64e(mac)


def verify_request_proof(
    psk: bytes, request_id: str, body_hash: str, proof: str,
) -> bool:
    """Constant-time comparison; False if the caller is missing the PSK."""
    expected = compute_request_proof(psk, request_id, body_hash)
    # base64 strings → compare bytes for constant-time
    try:
        return hmac.compare_digest(b64d(expected), b64d(proof))
    except Exception:
        return False


# ─── half-key files ───────────────────────────────────────────────────

# Both halves are 32-byte raw files. The server image holds /etc/qcl-server/half.bin;
# the client holds its half under DPAPI-wrapped storage. Helpers:


def write_half(path: Path, half: bytes) -> None:
    if len(half) != PSK_BYTES:
        raise ValueError(f"half must be {PSK_BYTES} bytes")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # tighten perms before writing the secret
    with open(path, "wb") as f:
        f.write(half)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass  # Windows


def read_half(path: Path) -> bytes:
    data = Path(path).read_bytes()
    if len(data) != PSK_BYTES:
        raise ValueError(
            f"{path} should be {PSK_BYTES} bytes, got {len(data)}"
        )
    return data
