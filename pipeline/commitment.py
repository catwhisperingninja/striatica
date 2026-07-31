"""Cryptographic commitment for sensitive semantic labels.

The point: keep striatica open-source and reproducible while proving — later,
to a vetted reviewer — that you held *exactly* a given set of sensitive semantic
labels as of a certain date, WITHOUT publishing the labels themselves.

How it works (commit-then-reveal):

  commit  →  salt = 16 random bytes;  digest = SHA-256(salt || canonical(data)).
             You PUBLISH the digest (e.g. commit it to the public repo — git's
             own history timestamps it). You KEEP the salt and the data private.

  reveal  →  later, hand the reviewer the data file + the salt. They recompute
             the digest and confirm it matches what you published months ago.
             That proves the data is byte-for-byte what you committed, unchanged.

Why this and not "hash the labels for secrecy": a one-way hash of the labels
would destroy them for legitimate use too, and short labels are guessable by
enumeration. A commitment is the right primitive — it proves possession and
integrity without revealing, and the whole-file salt makes it non-enumerable.

Properties:
  - Binding: any change to the data changes the digest (SHA-256).
  - Hiding: the random salt means the published digest reveals nothing about
    the data, and two commitments of identical data look unrelated.

This module NEVER prints or logs the label content — only digests and salts.
Standard library only; no third-party crypto, nothing home-rolled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
from pathlib import Path
from typing import Any

from pipeline.banner import detail, error, warn

SALT_BYTES = 16


def canonical_bytes(obj: Any) -> bytes:
    """Deterministic bytes for ``obj`` — same content, same bytes, any machine.

    Sorted keys and compact separators make the digest independent of key order
    and whitespace; ``ensure_ascii=False`` + UTF-8 makes it stable across
    platforms (macOS / WSL2) for unicode label text.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def new_salt() -> bytes:
    """A fresh 16-byte random salt (the per-commitment hiding value)."""
    return secrets.token_bytes(SALT_BYTES)


def compute_commitment(data: Any, salt: bytes) -> str:
    """SHA-256 hex digest of ``salt || canonical_bytes(data)`` (salt prefixed).

    The salt length is fixed at ``SALT_BYTES`` and enforced here: with a
    constant-length salt the boundary between salt and data is unambiguous, so
    a chosen variable-length salt cannot equivocate two distinct data values
    onto one digest (the attack an independent adversarial test surfaced).
    """
    if len(salt) != SALT_BYTES:
        raise ValueError(f"salt must be {SALT_BYTES} bytes, got {len(salt)}")
    return hashlib.sha256(salt + canonical_bytes(data)).hexdigest()


def commit(data: Any) -> tuple[str, str]:
    """Commit ``data``. Returns ``(salt_hex, digest_hex)``.

    PUBLISH the digest; keep the salt (and the data) private until reveal.
    """
    salt = new_salt()
    return salt.hex(), compute_commitment(data, salt)


def verify(data: Any, salt_hex: str, digest_hex: str) -> bool:
    """True iff ``data`` under ``salt_hex`` reproduces ``digest_hex``.

    Constant-time comparison so a caller can't learn the digest by timing.
    Malformed input (non-hex or wrong-length salt, non-ASCII digest) raises at
    this API boundary — the CLI (``main``) catches it into a clean nonzero exit.
    """
    recomputed = compute_commitment(data, bytes.fromhex(salt_hex))
    return secrets.compare_digest(recomputed, digest_hex.strip().lower())


# ── File helpers ────────────────────────────────────────────────────

def load_committable(path: str | Path) -> Any:
    """Load a .json or .jsonl file into a canonicalizable object.

    .jsonl → a list of the parsed lines (line order is part of the
    commitment); .json → the parsed object. Blank lines in .jsonl are skipped.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def commit_file(path: str | Path) -> tuple[str, str]:
    """Commit the contents of ``path``. Returns ``(salt_hex, digest_hex)``."""
    return commit(load_committable(path))


def verify_file(path: str | Path, salt_hex: str, digest_hex: str) -> bool:
    """True iff ``path`` under ``salt_hex`` reproduces ``digest_hex``."""
    return verify(load_committable(path), salt_hex, digest_hex)


# ── CLI ─────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="commitment",
        description="Commit / verify a cryptographic commitment over a private "
        "semantic-label file (never reveals the labels).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("commit", help="Commit a private file; prints the digest "
                                      "to publish and the salt to keep secret.")
    c.add_argument("path", help="Path to the private .json/.jsonl file.")

    v = sub.add_parser("verify", help="Verify a file against a published digest.")
    v.add_argument("path", help="Path to the file to check.")
    v.add_argument("--salt", required=True, help="Salt hex from the commit step.")
    v.add_argument("--digest", required=True, help="Published digest hex.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.cmd == "commit":
            salt_hex, digest_hex = commit_file(args.path)
            detail(f"Committed {args.path} (labels not shown, not logged).")
            # Raw values on stdout, plainly, so they can be copied/piped.
            print(f"digest (PUBLISH this):   {digest_hex}")
            print(f"salt   (KEEP SECRET):    {salt_hex}")
            warn("Publish the digest (e.g. commit it). Store the salt privately "
                 "with the data — NEVER commit the salt or the data.")
            return 0

        ok = verify_file(args.path, args.salt, args.digest)
        if ok:
            detail("MATCH — the file reproduces the published commitment.")
            return 0
        error("NO MATCH — the file does not match the published digest/salt.")
        return 1
    except FileNotFoundError as e:
        error(str(e))
        return 1
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        error(f"Could not process input: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
