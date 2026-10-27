# Post-publication Benjamini-Hochberg correction

## Scope

Pull request #13 corrected the repository's Benjamini-Hochberg implementation to apply the full step-up rejection rule. Current `main` contains that correction. This is a code correction made after the accepted historical study; it is not a retroactive rewrite of the accepted evidence.

## Frozen evidence boundary

The accepted `statarb_hist_etf_wf_v4` artifacts remain unchanged and hash-pinned in their recorded lineage:

- accepted evidence HEAD: `d1ef6fd04ac3f6e48f20f9424a029e19064af619`
- study freeze commit: `28be77fb018d14098a9079cd7e880f456d308599`
- 2025 artifact: `results/historical_oos/statarb_hist_etf_wf_v4_holdout_2025.json`
- 2025 artifact SHA-256: `b1197d0fb5280b54849270cfadd50bd450458522bbf6e482a1870cec723b6d66`
- frozen dataset ID: `yf_stat_arb_etfs_daily_2015_2025_v1`
- canonical dataset SHA-256: `a4082398991400c171062ee4ff2b12dba87e84c4f0133462ed9ee2fbb13f9fa8`

No accepted result, configuration, manifest, or publication artifact was overwritten as part of the correction.

## What the correction changes—and what it does not

The corrected step-up logic can change which individual hypotheses are rejected in windows where the rejection boundary is crossed. Therefore, some pre-2024 selected-pair identities may differ under the corrected implementation.

The 2025 conclusion is unchanged: each of the three pure-2025 windows has zero FDR survivors, zero qualifying pairs, and zero trades. The correction does not create a 2025 Sharpe ratio, return, drawdown, or turnover value, and it does not convert the null result into evidence of defensive skill or foresight.

## Required impact-study protocol

Any impact evaluation must remain separate from baseline reproduction:

1. Restore the original frozen raw files and verify every file hash plus the canonical dataset hash before execution.
2. Run the accepted code and the accepted code plus only the BH correction in isolated worktrees.
3. Use a new experiment ID and a new output directory for the corrected-code impact run.
4. Compare window-level survivors, selected-pair identities, positions, and downstream metrics.
5. Preserve the accepted `statarb_hist_etf_wf_v4` artifacts without modification.

The impact study has not been executed because the original raw files are not available in this repository. Re-downloaded vendor data would be a data-drift diagnostic, not a strict reproduction of the frozen study.
