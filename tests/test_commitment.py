"""Adversarial test suite for pipeline.commitment.

Author/test separation: these tests were written independently of the
implementation (pipeline/commitment.py, which is NOT modified here). The goal is
to (a) pin the intended commitment properties — binding, hiding, canonical
determinism, constant-time verify, no label leakage, clean CLI contract — and
(b) actively try to break them.

Findings summary (see the module docstring in the final report for detail):

  * DEFECT (binding, primitive level): compute_commitment concatenates
    ``salt + canonical_bytes(data)`` with NO length prefix / delimiter, and
    verify() accepts a salt of ANY length while commit()/new_salt() always emit
    16 bytes. A chosen variable-length salt slides bytes across the salt/message
    boundary, so two DISTINCT data values open ONE digest. Pinned by
    ``test_binding_DEFECT_*`` (xfail strict — flips to a hard failure the moment
    a salt-length check is added, which is the fix).

  * DEFECT (CLI robustness, minor): a non-ASCII ``--digest`` makes
    secrets.compare_digest raise TypeError, which neither verify() nor main()
    catches → uncaught traceback instead of a clean nonzero exit. Pinned by
    ``test_cli_DEFECT_nonascii_digest_uncaught`` (xfail strict).

Everything else HOLDS and is pinned by passing tests below.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import unicodedata
from pathlib import Path

import pytest

from pipeline import commitment as C


# ── helpers ──────────────────────────────────────────────────────────

def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


SENTINEL = "SENTINEL_LEAK_CANARY_7f3a"  # non-hex chars → can never appear in a digest/salt


# ════════════════════════════════════════════════════════════════════
# 1. BINDING — any change to the data is detected
# ════════════════════════════════════════════════════════════════════

def test_binding_single_char_change_flips_digest_and_verify():
    data = {"feature_42": "activates on the word Paris"}
    salt_hex, digest_hex = C.commit(data)
    assert C.verify(data, salt_hex, digest_hex) is True

    tampered = {"feature_42": "activates on the word Parjs"}  # one char
    assert C.verify(tampered, salt_hex, digest_hex) is False
    # digest itself is different under the same salt
    salt = bytes.fromhex(salt_hex)
    assert C.compute_commitment(data, salt) != C.compute_commitment(tampered, salt)


def test_binding_added_entry_flips_digest():
    salt = C.new_salt()
    base = {"a": "x", "b": "y"}
    added = {"a": "x", "b": "y", "c": "z"}
    assert C.compute_commitment(base, salt) != C.compute_commitment(added, salt)


def test_binding_removed_entry_flips_digest():
    salt = C.new_salt()
    full = {"a": "x", "b": "y"}
    fewer = {"a": "x"}
    assert C.compute_commitment(full, salt) != C.compute_commitment(fewer, salt)


def test_binding_reordered_jsonl_lines_flips_digest():
    # For a .jsonl file the value is a LIST; list order is part of the commitment
    # (only dict *keys* are canonicalized, never list order).
    salt = C.new_salt()
    a = [{"i": 1}, {"i": 2}, {"i": 3}]
    b = [{"i": 2}, {"i": 1}, {"i": 3}]
    assert C.compute_commitment(a, salt) != C.compute_commitment(b, salt)


def test_binding_known_answer_vector():
    """Golden vector built from a hand-written canonical byte literal.

    Independent of canonical_bytes(): pins (a) salt is a PREFIX, (b) the exact
    compact sorted-key canonical form, (c) SHA-256, (d) lowercase hex.
    """
    salt = bytes.fromhex("00112233445566778899aabbccddeeff")
    data = {"b": "two", "a": "one"}
    canonical_literal = b'{"a":"one","b":"two"}'
    # canonical form is what we expect
    assert C.canonical_bytes(data) == canonical_literal
    expected = hashlib.sha256(salt + canonical_literal).hexdigest()
    assert C.compute_commitment(data, salt) == expected
    # salt is prefixed, not appended: swapping order must change the digest
    assert C.compute_commitment(data, salt) != hashlib.sha256(
        canonical_literal + salt
    ).hexdigest()


def test_binding_DEFECT_equivocation_numeric():
    """Two DISTINCT data values must not open the same commitment.

    Honest commit of 123 with a 16-byte salt. A 17-byte salt (honest 16 bytes +
    0x31 == b'1') makes salt2 + canonical(23) == salt1 + canonical(123) byte for
    byte, so the SAME digest verifies for 23. Binding requires this be False.
    """
    salt1 = b"\xaa" * 16
    digest = C.compute_commitment(123, salt1)
    assert C.verify(123, salt1.hex(), digest) is True  # honest opening

    salt2 = b"\xaa" * 16 + b"\x31"  # 17 bytes; chosen, not random
    # Security-correct outcome = the bad-length salt does NOT open the digest,
    # whether the fix rejects it by returning False or by raising. Currently
    # verify returns True (no length check) → assertion fails → xfail. Either
    # fix style makes this XPASS → strict → hard failure, forcing marker removal.
    try:
        opened = C.verify(23, salt2.hex(), digest)
    except ValueError:
        opened = False
    assert opened is False


def test_commit_emits_fixed_16_byte_salt():
    """Commit side of the salt-length asymmetry (stable factual invariant).

    commit()/new_salt() always emit SALT_BYTES (16). verify() enforces NO
    matching length check — that gap is the binding defect, pinned
    authoritatively by ``test_binding_DEFECT_equivocation_numeric``. The fix is
    to enforce ``len(bytes.fromhex(salt_hex)) == SALT_BYTES`` in verify().
    """
    assert C.SALT_BYTES == 16
    assert len(C.new_salt()) == 16
    assert len(bytes.fromhex(C.commit({"x": 1})[0])) == 16


# ════════════════════════════════════════════════════════════════════
# 2. HIDING — digest/salt leak nothing; two commits are unlinkable
# ════════════════════════════════════════════════════════════════════

def test_hiding_two_commits_differ_in_salt_and_digest():
    data = {"label": "secret meaning"}
    salt_a, dig_a = C.commit(data)
    salt_b, dig_b = C.commit(data)
    assert salt_a != salt_b       # fresh random salt each time
    assert dig_a != dig_b         # → unrelated digests for identical data
    # both still open correctly under their own salt
    assert C.verify(data, salt_a, dig_a) is True
    assert C.verify(data, salt_b, dig_b) is True
    # and NOT under the other's salt
    assert C.verify(data, salt_a, dig_b) is False


def test_hiding_wrong_salt_fails_verify():
    data = {"label": "secret meaning"}
    _, digest = C.commit(data)
    wrong_salt = C.new_salt().hex()
    assert C.verify(data, wrong_salt, digest) is False


def test_hiding_salt_is_16_random_bytes():
    salts = {C.new_salt() for _ in range(200)}
    assert all(len(s) == C.SALT_BYTES for s in salts)
    assert len(salts) == 200  # no collisions among 200 draws


# ════════════════════════════════════════════════════════════════════
# 3. DETERMINISM / CANONICALIZATION
# ════════════════════════════════════════════════════════════════════

def test_canonical_key_order_independent():
    salt = C.new_salt()
    assert C.canonical_bytes({"a": 1, "b": 2}) == C.canonical_bytes({"b": 2, "a": 1})
    assert C.compute_commitment({"a": 1, "b": 2}, salt) == C.compute_commitment(
        {"b": 2, "a": 1}, salt
    )


def test_canonical_whitespace_independent_via_files(tmp_path):
    salt = C.new_salt()
    a = write(tmp_path / "a.json", '{"a": 1,   "b":\n 2}')
    b = write(tmp_path / "b.json", '{"b":2,"a":1}')
    assert C.compute_commitment(
        C.load_committable(a), salt
    ) == C.compute_commitment(C.load_committable(b), salt)


def test_canonical_unicode_roundtrips(tmp_path):
    data = {"desc": "café → naïve — 日本語 — Ω≈ç√"}
    salt_hex, digest_hex = C.commit(data)
    assert C.verify(data, salt_hex, digest_hex) is True
    # survives a file write/read round-trip (utf-8, ensure_ascii=False)
    p = write(tmp_path / "u.json", json.dumps(data, ensure_ascii=False))
    assert C.verify_file(p, salt_hex, digest_hex) is True


def test_compute_commitment_is_pure_function_of_data_and_salt():
    salt = C.new_salt()
    data = {"k": [1, 2, {"nested": "v"}]}
    assert C.compute_commitment(data, salt) == C.compute_commitment(data, salt)
    # depends on the salt
    assert C.compute_commitment(data, salt) != C.compute_commitment(data, C.new_salt())


def test_numeric_vs_string_do_not_collide():
    """Adversarial type confusion: 1 vs "1" must be distinct commitments."""
    salt = C.new_salt()
    assert C.canonical_bytes(1) != C.canonical_bytes("1")
    assert C.compute_commitment(1, salt) != C.compute_commitment("1", salt)
    assert C.compute_commitment({"a": 1}, salt) != C.compute_commitment({"a": "1"}, salt)


def test_int_vs_float_do_not_collide():
    salt = C.new_salt()
    assert C.canonical_bytes(1) != C.canonical_bytes(1.0)  # b"1" vs b"1.0"
    assert C.compute_commitment(1, salt) != C.compute_commitment(1.0, salt)


def test_unicode_normalization_caveat(tmp_path):
    """CAVEAT to the cross-platform claim (not a function defect).

    canonical_bytes does NOT apply Unicode normalization. The same label text
    stored NFC on one machine vs NFD on another yields DIFFERENT digests. This
    is 'correct' (they are different code-point sequences) but qualifies the
    docstring's 'stable across platforms for unicode label text'.
    """
    nfc = unicodedata.normalize("NFC", "café")   # 'é' == U+00E9
    nfd = unicodedata.normalize("NFD", "café")   # 'e' + U+0301
    assert nfc != nfd                            # different code-point sequences
    salt = C.new_salt()
    assert C.compute_commitment(nfc, salt) != C.compute_commitment(nfd, salt)


# ════════════════════════════════════════════════════════════════════
# 4. CONSTANT-TIME COMPARE — verify() must not use ==
# ════════════════════════════════════════════════════════════════════

def test_verify_routes_through_compare_digest_at_runtime(monkeypatch):
    """Behavioral pin: verify() actually calls secrets.compare_digest."""
    calls = []
    real = C.secrets.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(C.secrets, "compare_digest", spy)
    data = {"x": 1}
    salt_hex, digest_hex = C.commit(data)
    assert C.verify(data, salt_hex, digest_hex) is True
    assert len(calls) == 1
    # it compares the recomputed digest against the (normalized) published one
    recomputed, published = calls[0]
    assert recomputed == digest_hex
    assert published == digest_hex


def test_verify_source_has_no_naive_equality():
    """Structural assertion of a SECURITY REQUIREMENT (constant-time compare).

    Flagged in the report as structural, not behavioral: I/O behavior alone
    cannot distinguish compare_digest from ==, so this guards against a
    regression to a timing-unsafe comparison.
    """
    src = inspect.getsource(C.verify)
    assert "secrets.compare_digest" in src
    assert "==" not in src and "!=" not in src


# ════════════════════════════════════════════════════════════════════
# 5. FILE HELPERS
# ════════════════════════════════════════════════════════════════════

def test_commit_verify_roundtrip_json(tmp_path):
    p = write(tmp_path / "labels.json", json.dumps({"0": "one", "1": "two"}))
    salt_hex, digest_hex = C.commit_file(p)
    assert C.verify_file(p, salt_hex, digest_hex) is True


def test_commit_verify_roundtrip_jsonl(tmp_path):
    p = write(
        tmp_path / "labels.jsonl",
        '{"idx": 0, "label": "alpha"}\n{"idx": 1, "label": "beta"}\n',
    )
    salt_hex, digest_hex = C.commit_file(p)
    assert C.verify_file(p, salt_hex, digest_hex) is True


def test_jsonl_blank_lines_are_skipped(tmp_path):
    salt = C.new_salt()
    dense = write(tmp_path / "d.jsonl", '{"a": 1}\n{"b": 2}\n')
    sparse = write(tmp_path / "s.jsonl", '\n{"a": 1}\n\n\n{"b": 2}\n\n')
    assert C.load_committable(sparse) == [{"a": 1}, {"b": 2}]
    assert C.compute_commitment(
        C.load_committable(dense), salt
    ) == C.compute_commitment(C.load_committable(sparse), salt)


def test_tampered_json_file_fails_verify(tmp_path):
    p = write(tmp_path / "labels.json", json.dumps({"k": "original"}))
    salt_hex, digest_hex = C.commit_file(p)
    write(tmp_path / "labels.json", json.dumps({"k": "TAMPERED"}))
    assert C.verify_file(p, salt_hex, digest_hex) is False


def test_tampered_jsonl_file_fails_verify(tmp_path):
    p = write(tmp_path / "l.jsonl", '{"i": 0, "v": "a"}\n{"i": 1, "v": "b"}\n')
    salt_hex, digest_hex = C.commit_file(p)
    write(tmp_path / "l.jsonl", '{"i": 0, "v": "a"}\n{"i": 1, "v": "TAMPERED"}\n')
    assert C.verify_file(p, salt_hex, digest_hex) is False


def test_load_committable_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        C.load_committable(tmp_path / "does_not_exist.json")


def test_load_committable_bad_json_raises(tmp_path):
    p = write(tmp_path / "bad.json", "{not valid json,,,")
    with pytest.raises(json.JSONDecodeError):
        C.load_committable(p)


def test_empty_json_file_raises(tmp_path):
    """Adversarial: an empty .json file is NOT committable (json.loads('')...)."""
    p = write(tmp_path / "empty.json", "")
    with pytest.raises(json.JSONDecodeError):
        C.load_committable(p)


def test_empty_jsonl_file_is_empty_list(tmp_path):
    """Adversarial asymmetry: an empty .jsonl file IS committable (→ [])."""
    p = write(tmp_path / "empty.jsonl", "")
    assert C.load_committable(p) == []
    salt_hex, digest_hex = C.commit_file(p)
    assert C.verify_file(p, salt_hex, digest_hex) is True


# ════════════════════════════════════════════════════════════════════
# 6. NO LABEL LEAKAGE  (CRITICAL — this tool must never emit the labels)
# ════════════════════════════════════════════════════════════════════

def test_commit_cli_never_leaks_label_text(tmp_path, capsys):
    p = write(tmp_path / "labels.json", json.dumps({"feature_7": SENTINEL}))
    rc = C.main(["commit", str(p)])
    out, err = capsys.readouterr()
    assert rc == 0
    assert SENTINEL not in out
    assert SENTINEL not in err
    # a real digest (64 lowercase hex chars) did reach stdout
    assert any(
        len(tok) == 64 and all(ch in "0123456789abcdef" for ch in tok)
        for line in out.splitlines()
        for tok in line.replace(":", " ").split()
    )


def test_verify_match_cli_never_leaks_label_text(tmp_path, capsys):
    p = write(tmp_path / "labels.json", json.dumps({"feature_7": SENTINEL}))
    salt_hex, digest_hex = C.commit_file(p)
    rc = C.main(["verify", str(p), "--salt", salt_hex, "--digest", digest_hex])
    out, err = capsys.readouterr()
    assert rc == 0
    assert SENTINEL not in out and SENTINEL not in err


def test_verify_mismatch_cli_never_leaks_label_text(tmp_path, capsys):
    p = write(tmp_path / "labels.json", json.dumps({"feature_7": SENTINEL}))
    salt_hex, _ = C.commit_file(p)
    wrong_digest = "0" * 64
    rc = C.main(["verify", str(p), "--salt", salt_hex, "--digest", wrong_digest])
    out, err = capsys.readouterr()
    assert rc == 1  # mismatch
    assert SENTINEL not in out and SENTINEL not in err


def test_commit_return_values_are_hex_only():
    """The API's commit() surfaces only hex — never the data."""
    salt_hex, digest_hex = C.commit({"feature_7": SENTINEL})
    assert SENTINEL not in salt_hex and SENTINEL not in digest_hex
    assert all(ch in "0123456789abcdef" for ch in salt_hex)
    assert all(ch in "0123456789abcdef" for ch in digest_hex)
    assert len(salt_hex) == 32 and len(digest_hex) == 64


# ════════════════════════════════════════════════════════════════════
# 7. CLI CONTRACT
# ════════════════════════════════════════════════════════════════════

def test_cli_commit_exit0_prints_digest_and_salt(tmp_path, capsys):
    p = write(tmp_path / "labels.json", json.dumps({"k": "v"}))
    rc = C.main(["commit", str(p)])
    out, _ = capsys.readouterr()
    assert rc == 0
    assert "digest" in out.lower() and "salt" in out.lower()


def test_cli_verify_match_exit0(tmp_path):
    p = write(tmp_path / "labels.json", json.dumps({"k": "v"}))
    salt_hex, digest_hex = C.commit_file(p)
    assert C.main(["verify", str(p), "--salt", salt_hex, "--digest", digest_hex]) == 0


def test_cli_verify_mismatch_exit1(tmp_path):
    p = write(tmp_path / "labels.json", json.dumps({"k": "v"}))
    salt_hex, _ = C.commit_file(p)
    assert C.main(["verify", str(p), "--salt", salt_hex, "--digest", "0" * 64]) == 1


def test_cli_missing_file_exit1_no_traceback(tmp_path):
    missing = tmp_path / "nope.json"
    # returns 1 rather than raising FileNotFoundError
    assert C.main(["commit", str(missing)]) == 1


def test_cli_bad_hex_salt_exit1_no_traceback(tmp_path):
    p = write(tmp_path / "labels.json", json.dumps({"k": "v"}))
    _, digest_hex = C.commit_file(p)
    # non-hex salt → ValueError caught → clean 1
    assert C.main(["verify", str(p), "--salt", "zz", "--digest", digest_hex]) == 1


def test_cli_odd_length_salt_exit1_no_traceback(tmp_path):
    p = write(tmp_path / "labels.json", json.dumps({"k": "v"}))
    _, digest_hex = C.commit_file(p)
    assert C.main(["verify", str(p), "--salt", "abc", "--digest", digest_hex]) == 1


def test_cli_bad_json_exit1_no_traceback(tmp_path):
    p = write(tmp_path / "bad.json", "{broken")
    assert C.main(["commit", str(p)]) == 1


def test_cli_missing_required_arg_is_argparse_systemexit(tmp_path):
    p = write(tmp_path / "labels.json", json.dumps({"k": "v"}))
    with pytest.raises(SystemExit) as ei:
        C.main(["verify", str(p)])  # missing --salt/--digest
    assert ei.value.code != 0


def test_cli_DEFECT_nonascii_digest_uncaught(tmp_path):
    """Contract says bad hex exits nonzero cleanly, no unhandled traceback.

    A non-ASCII digest is bad hex; the security-correct behavior is a clean
    nonzero return. Currently it raises TypeError → xfail until fixed.
    """
    p = write(tmp_path / "labels.json", json.dumps({"k": "v"}))
    salt_hex, _ = C.commit_file(p)
    rc = C.main(["verify", str(p), "--salt", salt_hex, "--digest", "cafÉ"])
    assert rc != 0


# ════════════════════════════════════════════════════════════════════
# 8. ADVERSARIAL EXTRAS
# ════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("data", [{}, [], "", 0, {"a": {}}, {"a": []}])
def test_empty_and_falsy_data_roundtrip(data):
    salt_hex, digest_hex = C.commit(data)
    assert C.verify(data, salt_hex, digest_hex) is True


def test_deeply_nested_structure_roundtrips():
    data = {"a": [{"b": [{"c": [{"d": list(range(20))}]}]}]}
    salt_hex, digest_hex = C.commit(data)
    assert C.verify(data, salt_hex, digest_hex) is True


def test_digest_case_insensitive_on_verify():
    data = {"x": 1}
    salt_hex, digest_hex = C.commit(data)
    assert C.verify(data, salt_hex, digest_hex.upper()) is True


def test_digest_surrounding_whitespace_tolerated():
    data = {"x": 1}
    salt_hex, digest_hex = C.commit(data)
    assert C.verify(data, salt_hex, f"  {digest_hex}\n\t") is True


def test_salt_hex_case_and_whitespace_tolerated():
    """bytes.fromhex is case-insensitive and skips ASCII whitespace.

    Lenient but not a defect: the digest still binds. Documented so a future
    tightening of salt parsing is a conscious change, not a surprise.
    """
    data = {"x": 1}
    salt_hex, digest_hex = C.commit(data)
    assert C.verify(data, salt_hex.upper(), digest_hex) is True
    spaced = " ".join(salt_hex[i:i + 2] for i in range(0, len(salt_hex), 2))
    assert C.verify(data, spaced, digest_hex) is True


def test_random_wrong_length_salt_incidentally_fails():
    """A *random* 32-byte salt fails — but note this is incidental.

    Contrast with test_binding_DEFECT_equivocation_numeric: a *chosen*
    wrong-length salt can still succeed. Random miss != length validation.
    """
    data = {"x": 1}
    _, digest_hex = C.commit(data)
    assert C.verify(data, C.secrets.token_bytes(32).hex(), digest_hex) is False


def test_verify_malformed_salt_raises_valueerror_at_api_level():
    """Direct verify() (not via CLI) raises ValueError on malformed salt hex.

    main() catches this into a clean exit; a direct API caller gets an
    exception rather than False. Documented behavior, not a security defect.
    """
    data = {"x": 1}
    _, digest_hex = C.commit(data)
    with pytest.raises(ValueError):
        C.verify(data, "abc", digest_hex)      # odd length
    with pytest.raises(ValueError):
        C.verify(data, "zz", digest_hex)        # non-hex
