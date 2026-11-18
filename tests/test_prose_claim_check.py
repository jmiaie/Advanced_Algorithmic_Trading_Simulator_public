"""Tests for the supplementary prose-level claim checker (scripts/prose_claim_check.py).

Every test that mutates prose works on a tmp_path copy; the real tree is read only.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import prose_claim_check as pcc
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "prose_claim_check.py"
FALSE_DECIMAL = "\n\nThe 2025 run gave a hedge ratio of 7.4321 and a p-value of 0.0312.\n"
FALSE_COUNT = "\n\nAcross all buckets the pipeline ran 2,951 candidate tests over 41 windows.\n"


@pytest.fixture()
def pack_copy(tmp_path: Path) -> Path:
    """A tmp copy of the pack prose plus every artifact the checker reads."""
    dest = tmp_path / "repo"
    for src in [*pcc.source_paths(REPO), *(REPO / rel for rel in pcc.PROSE_DOCS)]:
        target = dest / src.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    return dest


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
    )


def append(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")


def test_committed_pack_passes() -> None:
    res = run_cli()
    assert res.returncode == 0, res.stdout + res.stderr
    assert res.stdout.startswith("prose claim check OK:")
    assert "supported by artifacts" in res.stdout and "allowlisted" in res.stdout


def test_committed_pack_has_small_allowlist_and_many_numbers() -> None:
    rep = pcc.check(REPO, pcc.DEFAULT_ALLOWLIST)
    assert rep.extracted >= 100
    assert rep.allowlisted <= 40
    assert rep.unsupported == []
    assert rep.stale == []


@pytest.mark.parametrize("doc", pcc.PROSE_DOCS)
@pytest.mark.parametrize("claim", [FALSE_DECIMAL, FALSE_COUNT], ids=["decimal", "count"])
def test_false_claim_appended_to_each_doc_fails(
    pack_copy: Path, doc: str, claim: str, capsys: pytest.CaptureFixture[str]
) -> None:
    append(pack_copy, doc, claim)
    assert pcc.main(["--root", str(pack_copy)]) == 1
    out = capsys.readouterr().out
    assert "FAILED" in out
    assert any(line.startswith(f"{doc}:") for line in out.splitlines())


def test_altering_one_of_many_occurrences_fails(
    pack_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    doc = f"{pcc.PACK}/TECHNICAL-PAPER.md"
    path = pack_copy / doc
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    hits = [i for i, ln in enumerate(lines) if "1,724" in ln]
    assert len(hits) >= 2, "premise: the value occurs more than once"
    lines[hits[0]] = lines[hits[0]].replace("1,724", "1,725", 1)
    path.write_text("".join(lines), encoding="utf-8")
    assert pcc.main(["--root", str(pack_copy)]) == 1
    flagged = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith(doc)]
    assert flagged == [f"{doc}:{hits[0] + 1} 1,725"]


def test_sign_flip_and_rounding_drift_fail(pack_copy: Path) -> None:
    doc = f"{pcc.PACK}/TECHNICAL-PAPER.md"
    path = pack_copy / doc
    text = path.read_text(encoding="utf-8")
    assert "−0.1499" in text
    path.write_text(text.replace("−0.1499", "+0.1499", 1), encoding="utf-8")
    assert pcc.main(["--root", str(pack_copy)]) == 1


def test_allowlisted_token_passes_but_only_up_to_count(
    pack_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    doc = pcc.PROSE_DOCS[1]
    append(pack_copy, doc, "\n\nA literature value of 7.4321 is quoted here.\n")
    allow = tmp_path / "allow.json"
    entry = {"file": doc, "token": "7.4321", "count": 1, "reason": "literature value (test)"}
    allow.write_text(json.dumps({"entries": [entry]}), encoding="utf-8")
    assert pcc.main(["--root", str(pack_copy), "--allowlist", str(allow)]) == 0
    assert "1 allowlisted" in capsys.readouterr().out
    # without the allowlist the same token fails
    empty = tmp_path / "empty.json"
    empty.write_text('{"entries": []}', encoding="utf-8")
    assert pcc.main(["--root", str(pack_copy), "--allowlist", str(empty)]) == 1
    # a second occurrence exceeds the allowlisted count
    append(pack_copy, doc, "\nAnd again 7.4321.\n")
    assert pcc.main(["--root", str(pack_copy), "--allowlist", str(allow)]) == 1


def test_allowlist_entry_requires_reason(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"entries": [{"file": "x", "token": "1.5"}]}), encoding="utf-8")
    res = run_cli("--allowlist", str(bad))
    assert res.returncode == 2
    assert "Traceback" not in res.stderr


def test_unknown_root_fails_cleanly(tmp_path: Path) -> None:
    for root in (tmp_path / "does-not-exist", tmp_path):
        res = run_cli("--root", str(root))
        assert res.returncode == 2
        assert "Traceback" not in res.stderr
        assert "ERROR" in res.stderr


def test_extraction_ignores_dates_hashes_sections_and_identifiers() -> None:
    text = (
        "### 10.4 Heading\nSee §6 and C-14 on 2026-09-17T03:44:34Z with 5768fd10b63c0436 "
        "in statarb_hist_etf_wf_v4_holdout_2025 using Python 3.11+.\n"
        "Real: 0.0312, 5%, 1,724 tests, 32 windows and −0.1499.\n"
    )
    got = [t.text for t in pcc.extract_tokens(text, "f.md")]
    assert got == ["0.0312", "5%", "1,724", "32", "−0.1499"]


def test_rounding_and_percent_matching() -> None:
    sup = pcc.Support([0.004090201526158799, 0.0023, 1724.0])

    def tok(s: str) -> pcc.Token:
        (t,) = pcc.extract_tokens(s, "f.md")
        return t

    assert sup.matches(tok("+0.0040902"))
    assert not sup.matches(tok("+0.0040903"))
    assert sup.matches(tok("0.23%"))
    assert sup.matches(tok("1,724"))
    assert not sup.matches(tok("1,725"))
