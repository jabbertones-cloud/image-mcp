"""Passphrase-encrypted at-rest storage.

NOTE — synced from D:\\App Dev\\QwenCharLoRA\\qwen_char_lora\\secure_store.py.
If you fix a bug here, fix it there too.

Use this to write secrets like the client PSK half to disk under a passphrase
that's never persisted. The same primitive can wrap any bytes (private keys,
config blobs, etc).

KDF:    Argon2id (libsodium reference; parameter pre-set ``MODERATE``)
AEAD:   XSalsa20-Poly1305 via NaCl ``SecretBox``

File format (v1):

  +----+-----+----+----------+--------------+----------+
  | 4B | 1B  | 1B | 16B salt | 32B + 24B    | tag (16) |
  | M  | ver | op |   salt   | enc(nonce||  | (Poly)   |
  |QCL1|  1  | 1  |          |   ciphertext)|          |
  +----+-----+----+----------+--------------+----------+

  M       magic bytes "QCL1"
  ver     format version (1)
  op      opslimit indicator (1 = MODERATE, 2 = SENSITIVE; we use 1)
  salt    Argon2 salt (16 bytes)
  rest    SecretBox.encrypt(plaintext) — nonce prefix included

We do NOT store the passphrase. Wrong-passphrase decrypt fails with a
clear DecryptionError; we surface that to the user as "wrong passphrase".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nacl.exceptions import CryptoError
from nacl.pwhash import argon2id
from nacl.secret import SecretBox
from nacl.utils import random as nacl_random


_MAGIC   = b"QCL1"
_VERSION = 1
_OPS_MODERATE = 1
_SALT_LEN = argon2id.SALTBYTES        # 16 in libsodium
_KEY_LEN  = SecretBox.KEY_SIZE        # 32


class DecryptionError(Exception):
    """Raised on tampered ciphertext or wrong passphrase."""


def _kdf(passphrase: str, salt: bytes, op_tag: int = _OPS_MODERATE) -> bytes:
    if op_tag == _OPS_MODERATE:
        ops = argon2id.OPSLIMIT_MODERATE
        mem = argon2id.MEMLIMIT_MODERATE
    else:
        ops = argon2id.OPSLIMIT_SENSITIVE
        mem = argon2id.MEMLIMIT_SENSITIVE
    return argon2id.kdf(_KEY_LEN, passphrase.encode("utf-8"), salt,
                        opslimit=ops, memlimit=mem)


def encrypt_with_passphrase(plaintext: bytes, passphrase: str) -> bytes:
    if not passphrase:
        raise ValueError("empty passphrase rejected")
    salt = nacl_random(_SALT_LEN)
    key  = _kdf(passphrase, salt)
    box  = SecretBox(key)
    ct   = box.encrypt(plaintext)            # 24-byte nonce prefix included
    return _MAGIC + bytes([_VERSION, _OPS_MODERATE]) + salt + ct


def decrypt_with_passphrase(blob: bytes, passphrase: str) -> bytes:
    if len(blob) < len(_MAGIC) + 2 + _SALT_LEN + SecretBox.NONCE_SIZE:
        raise DecryptionError("ciphertext too short")
    if blob[:4] != _MAGIC:
        raise DecryptionError("not a QCL secret store file")
    ver, op_tag = blob[4], blob[5]
    if ver != _VERSION:
        raise DecryptionError(f"unsupported store version {ver}")
    salt = blob[6 : 6 + _SALT_LEN]
    rest = blob[6 + _SALT_LEN :]
    key  = _kdf(passphrase, salt, op_tag=op_tag)
    try:
        return SecretBox(key).decrypt(rest)
    except CryptoError as e:
        raise DecryptionError("wrong passphrase or corrupted file") from e


@dataclass
class SecretFile:
    """High-level helper: encrypted file on disk, in-memory plaintext only
    when ``open`` is called. Always wrap reads/writes in a ``with`` block."""
    path: Path

    def write(self, plaintext: bytes, passphrase: str) -> None:
        blob = encrypt_with_passphrase(plaintext, passphrase)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # write atomically — tmp + rename — so a crash mid-write doesn't
        # leave a half-written file
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_bytes(blob)
        tmp.replace(self.path)
        try: self.path.chmod(0o600)
        except OSError: pass

    def read(self, passphrase: str) -> bytes:
        return decrypt_with_passphrase(self.path.read_bytes(), passphrase)


# ─── passphrase strength check ────────────────────────────────────────


# Conservative defaults. The CLI surfaces the reason if a passphrase is rejected.
MIN_PASSPHRASE_LEN = 12


def check_passphrase(p: str) -> list[str]:
    """Return a list of human-readable reasons the passphrase is weak.
    Empty list = acceptable."""
    issues: list[str] = []
    if len(p) < MIN_PASSPHRASE_LEN:
        issues.append(f"shorter than {MIN_PASSPHRASE_LEN} characters")
    classes = (
        any(c.islower() for c in p),
        any(c.isupper() for c in p),
        any(c.isdigit() for c in p),
        any(not c.isalnum() for c in p),
    )
    if sum(classes) < 3:
        issues.append("uses fewer than 3 of: lowercase, uppercase, digit, symbol")
    if p.isdigit():
        issues.append("digits only")
    return issues
