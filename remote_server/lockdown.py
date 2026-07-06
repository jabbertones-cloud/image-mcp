"""Process-level hardening — applied once at startup.

Best-effort. Logs what was and wasn't applied; never crashes the server
because a lockdown failed. Each routine returns True on success, False
on best-effort skip (non-Linux platform, missing privilege, etc.).

What we do:
  * PR_SET_DUMPABLE = 0      — no core dump on SIGSEGV / abort. Removes the
                                'dump-then-grep' attack on a crashed process.
                                Also makes /proc/PID/mem read-restricted to root.
  * PR_SET_NO_NEW_PRIVS = 1  — even if we exec a setuid binary, no privilege
                                gain. Defends against escalation post-compromise.
  * setrlimit RLIMIT_CORE = 0 — belt-and-brace: even if PR_SET_DUMPABLE is
                                bypassed, the OS still won't write the dump.
  * mlock secret buffers    — keep PSK + identity privkeys out of swap.
                                Best-effort; needs CAP_IPC_LOCK or rlimit room.
  * env-var scrub           — after lifespan reads REMOTE_*_B64, blank them
                                in os.environ so ``docker inspect`` returns
                                empty strings. Doesn't help if the daemon
                                already cached them, but reduces window.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import sys
from dataclasses import dataclass

try:
    import resource          # POSIX only — Windows dev hosts skip RLIMIT
except ImportError:
    resource = None          # type: ignore[assignment]

log = logging.getLogger(__name__)


@dataclass
class LockdownStatus:
    set_dumpable:    bool = False
    no_new_privs:    bool = False
    rlimit_core:     bool = False
    mlocks_done:     int  = 0
    env_scrubbed:    list[str] = None    # type: ignore[assignment]

    def __post_init__(self):
        if self.env_scrubbed is None:
            self.env_scrubbed = []


# ─── prctl ─────────────────────────────────────────────────────────


_PR_SET_DUMPABLE   = 4
_PR_SET_NO_NEW_PRIVS = 38


def _prctl(option: int, *args: int) -> bool:
    if sys.platform != "linux":
        return False
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6",
                          use_errno=True)
        # int prctl(int option, unsigned long arg2, unsigned long arg3,
        #           unsigned long arg4, unsigned long arg5);
        padded = list(args) + [0] * (4 - len(args))
        rc = libc.prctl(option, *padded)
        return rc == 0
    except Exception as e:
        log.warning(f"prctl({option}) failed: {e!r}")
        return False


def set_dumpable_false() -> bool:
    """Disable core dumps + make /proc/PID/mem read-restricted to root."""
    return _prctl(_PR_SET_DUMPABLE, 0)


def set_no_new_privs() -> bool:
    """Refuse to gain new privileges via exec (defence in depth)."""
    return _prctl(_PR_SET_NO_NEW_PRIVS, 1)


def set_rlimit_core_zero() -> bool:
    if resource is None:
        return False
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        return True
    except (ValueError, OSError) as e:
        log.warning(f"setrlimit(RLIMIT_CORE) failed: {e!r}")
        return False


# ─── mlock ─────────────────────────────────────────────────────────


def mlock_bytes(buf: bytes) -> bool:
    """Pin a bytes-like object's backing memory so it can't swap to disk.

    Best-effort. Failures (CAP_IPC_LOCK missing on Community-tier hosts,
    rlimit exhausted, etc.) are logged but not raised."""
    if sys.platform != "linux":
        return False
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6",
                          use_errno=True)
        # mlock takes (addr, len); we need a pointer into the Python object.
        # The id()-based trick is fragile across interpreters; PyNaCl exposes
        # sodium_mlock for the same purpose if you've got libsodium.
        try:
            import nacl.bindings as nbi
            nbi.sodium_mlock(buf)
            return True
        except Exception:
            pass
        # fallback: ctypes mlock on the buffer-protocol view
        addr = ctypes.addressof(ctypes.c_char.from_buffer_copy(buf))
        rc = libc.mlock(addr, len(buf))
        return rc == 0
    except Exception as e:
        log.warning(f"mlock failed: {e!r}")
        return False


# ─── env scrub ─────────────────────────────────────────────────────


def scrub_env(prefixes: list[str] | None = None) -> list[str]:
    """Clear any os.environ entry whose name starts with one of ``prefixes``.

    Defaults: ['REMOTE_SERVER_HALF_B64', 'REMOTE_CLIENT_HALF_B64',
              'REMOTE_SERVER_IDENTITY_B64', 'REMOTE_CLIENT_PEER_B64',
              'TS_AUTHKEY']

    Returns the list of names that were cleared. The values are gone from
    Python's view of the env. ``docker inspect`` may still show them if the
    Docker daemon snapshotted them at container creation — there's no fix
    for that short of unsetting them in the parent shell before docker run."""
    prefixes = prefixes or [
        "REMOTE_SERVER_HALF_B64",
        "REMOTE_CLIENT_HALF_B64",
        "REMOTE_SERVER_IDENTITY_B64",
        "REMOTE_CLIENT_PEER_B64",
        "TS_AUTHKEY",
    ]
    cleared: list[str] = []
    for k in list(os.environ):
        if any(k == p or k.startswith(p) for p in prefixes):
            # Overwrite first (best-effort: if the underlying envp array is
            # shared with libc strings the bytes may get scrubbed in place).
            # Windows refuses null bytes in env values — use a printable
            # filler instead. Then unset.
            try:
                old_len = len(os.environ[k])
                os.environ[k] = "X" * max(old_len, 32)
            except (OSError, ValueError):
                pass
            try: del os.environ[k]
            except KeyError: pass
            cleared.append(k)
    return cleared


# ─── apply-all entry point ─────────────────────────────────────────


def apply_lockdowns(*, secrets_to_mlock: list[bytes] | None = None) -> LockdownStatus:
    s = LockdownStatus()
    s.set_dumpable  = set_dumpable_false()
    s.no_new_privs  = set_no_new_privs()
    s.rlimit_core   = set_rlimit_core_zero()
    s.env_scrubbed  = scrub_env()
    if secrets_to_mlock:
        for buf in secrets_to_mlock:
            if mlock_bytes(buf):
                s.mlocks_done += 1
    log.info(
        f"lockdown: dumpable={'OFF' if s.set_dumpable else 'skip'} "
        f"no_new_privs={'ON' if s.no_new_privs else 'skip'} "
        f"core_rlimit={'0' if s.rlimit_core else 'skip'} "
        f"mlocks={s.mlocks_done} env_scrubbed={s.env_scrubbed}"
    )
    return s
