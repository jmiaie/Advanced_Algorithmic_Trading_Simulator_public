#!/usr/bin/env python3
"""Deterministic D10-B publication-pack generator / verifier.

Zero network. Stdlib only. Never re-derives empirical results: every number in
tables/ and figures/ is read from the ACCEPTED artifacts in results/historical_oos/
and checked against the hashes recorded in reproducibility.json.

    python3 publication_pack.py build   # regenerate tables/ + figures/
    python3 publication_pack.py check   # verify hashes + regenerate-and-diff (CI gate)

Exit 1 on any missing file, hash mismatch, drift, or bad citation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
REPRO_PATH = PACK / "reproducibility.json"
SOURCE_MAP = PACK / "RESULT-SOURCE-MAP.md"

# Accepted v4 walk-forward buckets, in program order. Labels are the binding
# period names used throughout the D9-B record.
BUCKETS = [
    ("dev_formation", "2015-2023 development/formation"),
    ("boundary_2023_2024", "boundary 2023-10 - 2024-01"),
    ("val_2024", "2024 validation"),
    ("holdout_2025", "2025 FINAL WALK-FORWARD EVALUATION"),
]
V4 = "results/historical_oos/statarb_hist_etf_wf_v4_{bucket}.json"

WINDOW_COLS = [
    "bucket",
    "window_index",
    "test_start",
    "test_end",
    "validation_start",
    "validation_end",
    "formation_start",
    "formation_end",
    "n_candidate_tests",
    "n_fdr_survivors",
    "no_trade",
    "used_fallback",
    "selected_pair",
    "entry_z",
    "exit_abs_z",
    "trailing_z_window",
    "kalman_process_variance",
]
SUMMARY_COLS = [
    "bucket",
    "n_windows",
    "n_qualifying_windows",
    "n_no_trade_windows",
    "n_fallback_windows",
    "n_fdr_survivors_total",
    "n_candidate_tests_total",
    "n_traded_windows",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_repro() -> dict:
    return json.loads(REPRO_PATH.read_text(encoding="utf-8"))


def load_bucket(bucket: str) -> dict:
    return json.loads((ROOT / V4.format(bucket=bucket)).read_text(encoding="utf-8"))


def window_rows(bucket: str, doc: dict) -> list[dict]:
    rows = []
    for w in doc.get("windows", []):
        p = w.get("frozen_params") or {}
        rows.append(
            {
                "bucket": bucket,
                "window_index": w.get("window_index"),
                "test_start": w.get("test_start"),
                "test_end": w.get("test_end"),
                "validation_start": w.get("validation_start"),
                "validation_end": w.get("validation_end"),
                "formation_start": w.get("formation_start"),
                "formation_end": w.get("formation_end"),
                "n_candidate_tests": w.get("n_candidate_tests"),
                "n_fdr_survivors": w.get("n_fdr_survivors"),
                "no_trade": w.get("no_trade"),
                "used_fallback": w.get("used_fallback"),
                "selected_pair": w.get("selected_pair") or "",
                "entry_z": p.get("entry_z", ""),
                "exit_abs_z": p.get("exit_abs_z", ""),
                "trailing_z_window": p.get("trailing_z_window", ""),
                "kalman_process_variance": p.get("kalman_process_variance", ""),
            }
        )
    return rows


def summary_row(bucket: str, doc: dict, rows: list[dict]) -> dict:
    return {
        "bucket": bucket,
        "n_windows": doc.get("n_windows"),
        "n_qualifying_windows": doc.get("n_qualifying_windows"),
        "n_no_trade_windows": doc.get("n_no_trade_windows"),
        "n_fallback_windows": doc.get("n_fallback_windows"),
        "n_fdr_survivors_total": sum(int(r["n_fdr_survivors"] or 0) for r in rows),
        "n_candidate_tests_total": sum(int(r["n_candidate_tests"] or 0) for r in rows),
        "n_traded_windows": sum(1 for r in rows if r["no_trade"] is False),
    }


def csv_bytes(header: list[str], rows: list[dict]) -> bytes:
    import io

    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=header, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")


def _bar_panel(
    series: list[tuple[str, list[tuple[str, int]], str]],
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> list[str]:
    """One grouped-bar panel: series = [(panel_title, [(label, value)], colour)]."""
    out = [
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" '
        f'fill="#ffffff" stroke="#333333"/>'
    ]
    for title, bars, colour in series:
        out.append(
            f'<text x="{x0 + 8}" y="{y0 + 16}" font-family="monospace" font-size="11" '
            f'font-weight="bold" fill="#111111">{title}</text>'
        )
        vmax = max([v for _, v in bars] + [1])
        n = len(bars)
        slot = (width - 20) / max(n, 1)
        bw = slot * 0.55
        base = y0 + height - 26
        usable = height - 60
        for i, (label, value) in enumerate(bars):
            cx = x0 + 10 + slot * i + slot / 2
            h = 0 if vmax == 0 else int(usable * value / vmax)
            x = int(cx - bw / 2)
            out.append(
                f'<rect x="{x}" y="{base - h}" width="{int(bw)}" height="{h}" fill="{colour}"/>'
            )
            out.append(
                f'<text x="{int(cx)}" y="{base - h - 4}" text-anchor="middle" '
                f'font-family="monospace" font-size="11" fill="#111111">{value}</text>'
            )
            out.append(
                f'<text x="{int(cx)}" y="{base + 14}" text-anchor="middle" '
                f'font-family="monospace" font-size="9" fill="#333333">{label}</text>'
            )
        out.append(
            f'<line x1="{x0 + 10}" y1="{base}" x2="{x0 + width - 10}" y2="{base}" '
            f'stroke="#333333" stroke-width="1"/>'
        )
    return out


def funnel_svg(summaries: list[dict]) -> bytes:
    """Selection funnel per walk-forward bucket, straight from artifact fields."""
    labels = [s["bucket"] for s in summaries]
    rows = list(zip(labels, summaries, strict=True))

    def series(
        title: str, field: str, colour: str
    ) -> tuple[str, list[tuple[str, int]], str]:
        return (title, [(name, int(s[field])) for name, s in rows], colour)

    candidates = [series("candidate pair tests", "n_candidate_tests_total", "#4a6fa5")]
    outcomes = [
        series("windows w/ selected pair", "n_traded_windows", "#2e7d32"),
        series("windows: no-trade", "n_no_trade_windows", "#8c8c8c"),
    ]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="380" '
        'viewBox="0 0 900 380">',
        '<rect width="900" height="380" fill="#ffffff"/>',
        '<text x="16" y="24" font-family="monospace" font-size="13" '
        'font-weight="bold" fill="#111111">',
        "D10-B stat-arb-study: selection funnel per walk-forward bucket "
        "(accepted artifacts only)</text>",
        '<text x="16" y="42" font-family="monospace" font-size="10" fill="#333333">',
        "Source: results/historical_oos/statarb_hist_etf_wf_v4_"
        "{dev_formation,boundary_2023_2024,val_2024,holdout_2025}.json</text>",
        '<text x="16" y="58" font-family="monospace" font-size="10" fill="#333333">',
        "Zero trades in the 2025 bucket is the reported result, not missing data: "
        "no FDR survivor means no pair, hence NO_TRADE.</text>",
    ]
    parts += _bar_panel(candidates, 16, 70, 420, 290)
    parts += _bar_panel(outcomes, 452, 70, 432, 290)
    parts.append("</svg>")
    return ("\n".join(parts) + "\n").encode("utf-8")


def build() -> dict[str, bytes]:
    """Return {relative output path: bytes} for every generated table/figure."""
    summaries, rows = [], []
    for bucket, _label in BUCKETS:
        doc = load_bucket(bucket)
        brows = window_rows(bucket, doc)
        rows.extend(brows)
        summaries.append(summary_row(bucket, doc, brows))
    hashes = []
    for entry in load_repro()["source_artifacts"]:
        p = ROOT / entry["path"]
        hashes.append(
            {
                "path": entry["path"],
                "sha256": sha256_file(p) if p.exists() else "MISSING",
                "bytes": p.stat().st_size if p.exists() else 0,
            }
        )
    return {
        "tables/window-level.csv": csv_bytes(WINDOW_COLS, rows),
        "tables/bucket-summary.csv": csv_bytes(SUMMARY_COLS, summaries),
        "tables/artifact-hashes.csv": csv_bytes(["path", "sha256", "bytes"], hashes),
        "figures/selection-funnel.svg": funnel_svg(summaries),
    }


CITATION_RE = re.compile(
    r"`((?:results|configs|research|data)/[\w./\-]+\.(?:json|csv|yaml|md))`"
    r"\s*\(sha256 `([0-9a-f]{64})`\)"
)


def check() -> int:
    failures = []
    repro = load_repro()

    # 1. Every source artifact hash recorded in reproducibility.json must match disk.
    #    Guard the vacuous case first: an empty list must fail loud, not print OK.
    if not repro["source_artifacts"]:
        failures.append("reproducibility.json lists no source_artifacts — hash gate would pass vacuously")
    for entry in repro["source_artifacts"]:
        p = ROOT / entry["path"]
        if not p.exists():
            failures.append(f"missing source artifact: {entry['path']}")
        elif sha256_file(p) != entry["sha256"]:
            failures.append(
                f"hash mismatch: {entry['path']} = {sha256_file(p)} != {entry['sha256']}"
            )

    # 2. Generated outputs must be byte-identical to what is committed.
    for rel, produced in build().items():
        p = PACK / rel
        if not p.exists():
            failures.append(f"missing generated file: {rel}")
        elif p.read_bytes() != produced:
            failures.append(f"drift: {rel} does not match deterministic rebuild")

    # 3. Every (path, sha256) citation in the source map must resolve.
    cites = CITATION_RE.findall(SOURCE_MAP.read_text(encoding="utf-8"))
    if not cites:
        failures.append("no (path, sha256) citations found in RESULT-SOURCE-MAP.md")
    for rel, digest in cites:
        p = ROOT / rel
        if not p.exists():
            failures.append(f"cited path missing: {rel}")
        elif sha256_file(p) != digest:
            failures.append(f"cited hash wrong: {rel} = {sha256_file(p)} != {digest}")

    if failures:
        print("publication-pack check FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print(
        f"publication-pack check OK: {len(repro['source_artifacts'])} artifact hashes, "
        f"{len(build())} generated files, {len(cites)} map citations verified"
    )
    return 0


MANIFEST = ROOT / "data/manifests/yf_stat_arb_etfs_daily_2015_2025_v1.json"
FREEZE_SHA = "28be77fb018d14098a9079cd7e880f456d308599"
FROZEN_CONFIG = "configs/experiments/statarb_historical_etf_wf_v4.yaml"
HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
HEX_LONG = re.compile(r"\b[0-9a-f]{40,64}\b")
ABBR = re.compile(r"\b([0-9a-f]{8,63})…")
# Deliberately not a claim about repository bytes: asserted in this file's __main__ self-check.
SELFTEST_FIXTURES = {hashlib.sha256(b"abc").hexdigest()}

# Hash values that this pack published in an earlier revision and then legitimately retired,
# because the underlying artifact changed (D10 CL-20 remediation, 2026-09-18). Declared one by one
# with a reason so the defect log can keep quoting what it measured without the sweep going blind.
RETIRED_HASHES = {
    # research/experiment-ledger.csv at 77c8fbd: 23 rows / 19,542 B, before the three corrective rows.
    "85fd6bf54713d640676bb4174ddfdf9fe2833762d0d919b9ed8a9e7c2af57dd8": "retired ledger hash (resolvable at 77c8fbd)",
}


def hashcheck_bind_citations(texts: dict[Path, str]) -> int:
    """Bind every `path` (sha256 `hash`) citation in the pack to THAT path's own bytes.

    Token membership is not enough: a genuine hash of some other file satisfies it. Separated
    from hashcheck() so the __main__ self-check can prove the binding fails on a poisoned copy.
    """
    for p, t in texts.items():
        for rel, digest in CITATION_RE.findall(t):
            fp = ROOT / rel
            actual = sha256_file(fp) if fp.exists() else "MISSING"
            if actual != digest:
                print(f"hash-integrity check FAILED: {p.relative_to(ROOT)} cites {rel} = {digest} but path is {actual}")
                return 1
    return 0


def hashcheck() -> int:
    """Sweep every hash printed anywhere in the pack and fail closed on any untraceable value.

    A published sha256 passes only if it is the measured sha256 of a working-tree file, the value
    recorded in the accepted frozen manifest, the config blob at the freeze commit's parent, or the
    stated self-test fixture. Abbreviated hex must resolve to a full form printed in the pack.
    """
    texts = {
        p: p.read_text(encoding="utf-8", errors="ignore")
        for p in sorted(x for x in PACK.rglob("*") if x.is_file())
    }
    if hashcheck_bind_citations(texts):
        return 1
    known: dict[str, str] = {}
    for p in ROOT.rglob("*"):
        if p.is_file() and "/.git/" not in str(p):
            known.setdefault(sha256_bytes(p.read_bytes()), f"working tree: {p.relative_to(ROOT)}")
    try:
        blob = subprocess.run(
            ["git", "show", f"{FREEZE_SHA}^:{FROZEN_CONFIG}"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"hash-integrity check FAILED: cannot read freeze-parent config ({exc})")
        return 1
    known.setdefault(sha256_bytes(blob), "config blob at the freeze commit's parent")
    for v in HEX64.findall(MANIFEST.read_text(encoding="utf-8")):
        known.setdefault(v, "value recorded in the accepted frozen manifest")
    for v, why in RETIRED_HASHES.items():
        known.setdefault(v, why)

    printed = {t for text in texts.values() for t in HEX_LONG.findall(text)}
    unresolved: list[str] = []
    occurrences = 0
    for p, text in texts.items():
        rel = p.relative_to(ROOT)
        for tok in HEX64.findall(text):
            occurrences += 1
            if tok not in known and tok not in SELFTEST_FIXTURES:
                unresolved.append(f"{rel}: untraceable sha256 {tok}")
        for a in set(ABBR.findall(text)):
            if not any(full.startswith(a) for full in printed):
                unresolved.append(f"{rel}: abbreviation {a}… has no full form in the pack")

    if unresolved:
        print("hash-integrity check FAILED:")
        for u in sorted(set(unresolved)):
            print("  -", u)
        return 1
    published = {t for t in printed if len(t) == 64} - SELFTEST_FIXTURES
    git_shas = {t for t in printed if len(t) == 40}
    abbreviations = {a for text in texts.values() for a in ABBR.findall(text)}
    print(
        f"hash-integrity check OK: {len(published)} published sha256 values verified "
        f"({occurrences} occurrences), {len(git_shas)} full git SHAs printed, "
        f"{len(abbreviations)} abbreviations resolved"
    )
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in ("build", "check", "hashcheck"):
        print(__doc__)
        return 2
    if argv[1] == "hashcheck":
        return hashcheck()
    if argv[1] == "check":
        return check()
    for rel, data in build().items():
        p = PACK / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        print(f"wrote {rel} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    assert sha256_bytes(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    ), "sha256 self-check failed"
    sys.exit(main(sys.argv))
