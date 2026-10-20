# Study A — power / null calibration of the stat-arb pair screen (frozen protocol)

Status: **frozen before the full simulation run** (commit 1 of the PR on branch
`research/power-calibration`; the push timestamp of that commit is the design freeze).
Any later change is recorded below under *Amendments*, dated, with its reason.

Scope: this is a calibration of the existing, frozen v4 design. It does not re-run, re-tune or
re-interpret the 2025 evaluation, and it changes no frozen artifact, config, data manifest, result
JSON, publication-pack file or source module. New files only: `research/power/`,
`tests/test_power_study.py`, `results/power/`.

## 1. Question

The published 2025 walk-forward evaluation (`configs/experiments/statarb_historical_etf_wf_v4.yaml`)
found **0 FDR survivors in 3 windows × 55 tests**. The technical paper cannot tell "no
cointegration" from "screen too strict", and records `validation_ols_sharpe = null` in all 32
windows (claim C-31). This study answers:

* **A1.** If a within-group pair were truly cointegrated with spread half-life *h*, what is the
  probability that the frozen screen admits it (power)? What is the largest half-life detectable
  with ≥ 80% power (MDE)? What false-survivor count and realised FDR does the screen produce?
* **A2.** Can the validation-grid selection path that populates `validation_ols_sharpe` ever fire
  on a strong synthetic cointegrated pair through the real walk-forward runner — i.e. is C-31 a
  property of the data or of a code path that cannot fire?

## 2. Frozen design parameters (read from the config and code, not re-chosen)

| Parameter | Value | Source |
|---|---|---|
| Formation window | 504 sessions | config `walk_forward.formation_size`; `wf_v4_orch.run_walk_forward_study` |
| Validation / test / step | 126 / 63 / 63 | config `walk_forward` |
| Universe | 19 ETFs, 4 groups (A: 4, B: 10, C: 3, D: 2) | config `universe.groups` |
| Candidate tests per window | 6 + 45 + 3 + 1 = **55** within-group pairs | `wf_v3_pairs.enumerate_group_candidates` |
| Pair test | Engle-Granger `statsmodels.tsa.stattools.coint(y, x)` defaults (constant, AIC lag selection), on **price levels**, y = lexicographically first symbol | `research.fit_static_ols_hedge_ratio` (research.py:122-143) |
| Multiple testing | Benjamini-Hochberg pooled over all 55 EG p-values, q = `fdr_alpha` = **0.05**, current step-up rule `rejected = adjusted <= alpha` | `research.benjamini_hochberg` (research.py:209-239); config `selection.fdr_alpha` |
| ADF admission | `adfuller(residual)` defaults on the formation OLS residual; admitted iff **p < 0.05** (strict) | wf_v3_pairs.py:99; config `selection.adf_alpha` |
| Survivor | `eligible = fdr_rejected & adf_passes`; `n_fdr_survivors` = count of eligible pairs | wf_v3_pairs.py:100-103 |
| Selection | at most one pair: lowest BH-adjusted p, then lower ADF p, then symbols | wf_v3_pairs.py:112-118 |
| Validation grid | entry_z {1.5, 2, 2.5} × exit_abs_z {0.25, 0.5, 0.75} × z-window {20, 40, 60}; BASE costs; static OLS net Sharpe; **≥ 10 validation trades** (`MIN_VALIDATION_TRADES`) | wf_v4_orch.py:47, 127-171 |

The simulation calls `stat_arb_engine.wf_v3_pairs.select_pair_within_groups` directly (the
production screen, which calls the production EG/ADF fit and BH). Design values are loaded at run
time from the frozen config (read-only) and asserted in `tests/test_power_study.py`.

## 3. A1 — screen power simulation

**Data-generating process (per replication).**

* 19 legs are independent geometric random walks over 504 business days:
  `P_t = 100 · exp(Σ σ z_i)`, z ~ N(0, 1), daily log-vol σ.
* *k* planted pairs, disjoint symbols so that no extra (transitive) cointegrated pair is created:
  k = 1: (XLB, XLE); k = 3: (XLB, XLE), (QQQ, SPY), (IEF, TLT). The first symbol of each is the
  production regressand and is replaced by `y = β · x + s`, x being the second symbol's random walk.
* Spread `s_t = φ s_{t−1} + ε_t`, `φ = 0.5^(1/h)`, stationary start `s_0 = ε_0 / √(1−φ²)`,
  ε ~ N(0, 1.0²) in price units (= 1% of the 100 start price), **fixed in every cell**.
  Assumption (rule 4: no committed price data exist in the repo): the spread innovation scale is
  set equal to the default leg's daily price vol. Varying σ with the spread scale held fixed is
  therefore the sensitivity to the leg-vol / spread-vol ratio (0.5, 1, 2).
* All other pairs, including those that share one planted leg with an independent leg, are nulls.

**Grid.**

| Block | σ | β | h (days) | k | reps per cell |
|---|---|---|---|---|---|
| Primary | 1% | 0.5, 1, 2 | 5, 10, 20, 40, 60, 120 | 1, 3 | 500 |
| σ sensitivity | 0.5%, 2% | 1 | 5, 10, 20, 40, 60, 120 | 1, 3 | 500 |
| Global null (k = 0) | 0.5%, 1%, 2% | — | — | 0 | 500 |

36 + 24 planted cells + 3 null cells; 500 replications per cell (Monte-Carlo SE of a power
estimate ≤ 2.2 pp; ≤ 1.3 pp for k = 3, which pools 1,500 planted-pair trials). The σ-sensitivity
block is run at β = 1 only by design (decided here, before any run, to keep the full run near
one hour on 4 cores).

**Seeds.** Master seed `20261020`. Null legs for (σ, rep):
`SeedSequence([20261020, round(σ·1e4), rep, 0])`. Spread innovations for planted slot j:
`SeedSequence([20261020, round(σ·1e4), rep, 1, j])`. Reps 0..499 in every cell. The same legs and
innovations are reused across h, β and k within a (σ, rep) — common random numbers, so differences
between cells are not Monte-Carlo noise in the legs.

**Implementation note (exactness).** Because the null legs are shared across cells, most of the 55
fits recur with identical inputs. `memoized_production_fit` wraps the production
`fit_static_ols_hedge_ratio` in an exact-input cache (key = bytes of both series and their index).
Every value used was computed by the production function on exactly those inputs; a unit test
asserts cached and uncached screen outputs are identical.

**Metrics (per cell).**

* **Power** = P(planted pair is a survivor) = planted survivors / (reps × k), with Wilson 95% CI.
  Also reported: P(all k planted survive), P(≥ 1 planted survives), P(selected pair is planted),
  BH-only power and ADF-only pass rate (to show which gate binds).
* **Expected false survivors** = mean number of non-planted survivors per window.
* **Realised FDR** = mean over reps of V / max(R, 1).
* **P(any survivor)** (for k = 0: the global-null false-alarm rate per window).
* **MDE** = largest grid half-life with power ≥ 0.80 (per σ, β, k, rule); "none" if no grid
  point reaches 0.80. The MDE is reported at grid resolution only (no interpolation).
* Size: BH at q = 0.05 as configured (the "5% size" of the shared protocol is the design's own q).

**Pre-fix rule sensitivity.** The same simulated windows are also scored under the PRE-FIX
(before 2026-10-14, commit a2a495e) per-rank BH rule `p_(k) <= α·k/m`, implemented locally and
labelled as such in `research/power/screen_power.py::bh_reject_prefix_per_rank`, combined with the
same ADF admission. `src/` is not modified.

## 4. A2 — end-to-end positive control

* **A2a.** A synthetic 19-symbol close panel over business days 2015-01-02 … 2025-12-31 (the
  config's data span), with one planted pair XLB = 1·XLE + s (σ = 1%, spread ε sd 1.0), written as
  per-symbol CSVs and run through the unmodified `scripts/run_statarb_wf_v4_study.py` with the
  frozen config, `--raw-dir` pointing to the synthetic CSVs and `--results-dir` / `--ledger`
  pointing to a temporary directory. Both modes: DEV/2024-validation (default) and
  `--allow-holdout`. Half-lives h = 2 (strong) and h = 20 (moderate); panel seed 0
  (`SeedSequence([20261020, 2, 0])`). Recorded per window: selected pair, `used_fallback`,
  `validation_ols_sharpe`, `validation_kalman_sharpe`, static-OLS BASE test trades and return.
* **A2b.** The step that sets `validation_ols_sharpe`
  (`wf_v4_orch._validation_grid_search_signal_params`) on one planted pair (504 + 126 + 63 bars,
  σ = 1%, β = 1), h ∈ {1, 2, 3, 5, 10, 20, 40}, 200 reps each (`SeedSequence([20261020, 3, rep])`):
  P(grid returns a selection), and the distribution of the maximum validation trade count over the
  27 grid points versus the 10-trade minimum.

## 5. Interpretation rules (written before any result)

1. **MDE headline** = MDE at k = 1, β = 1, σ = 1%, current rule. Classification:
   * MDE ≥ 60 days → the screen detects month-to-quarter reversion; a 0-survivor window is
     evidence against such relationships among the 55 tested pairs in that window.
   * 20 ≤ MDE < 60 → the screen detects roughly month-scale reversion only; slower relationships
     are not ruled out by a 0-survivor window.
   * MDE < 20 (or none) → a 0-survivor window rules out only very fast reversion and is
     uninformative about half-lives ≥ 20 days; "0 survivors" cannot distinguish "no
     cointegration" from "screen too strict" at typical pair horizons.
2. **What the 2025 null rules out.** For a relationship of half-life h present in a given 2025
   window, P(0 survivors among planted | h) ≈ 1 − power(h). The three 2025 formation windows
   overlap by 441 of 504 days, so they are **not** independent trials; no (1 − p)³ multiplication
   is claimed. The statement is made per window.
3. **Robustness.** The MDE is "robust to σ / β" if it is the same grid point across all σ and β
   cells for the same k; otherwise the range is reported as the headline's uncertainty.
4. **Did the BH correction matter for power?** "Material" iff |power_current − power_prefix| ≥
   0.05 in at least one primary cell; otherwise "immaterial for power".
5. **Null calibration.** Under the global null BH controls the FWER at q. The screen is called
   "anti-conservative" if P(any survivor | k = 0) > 0.05 + 2·MC SE in any σ; otherwise "at or below
   nominal". Realised FDR in planted cells is compared with q = 0.05 the same way.
6. **A2 verdict.** The grid path "can fire" if, in A2a DEV mode at h = 2, the planted pair is
   selected and `validation_ols_sharpe` is populated in ≥ 1 window. If not, it is reported as a
   FINDING with the code path (file:line) and evidence. Independently of that verdict, any path
   that sets `validation_ols_sharpe` to null by construction is reported with file:line. A2b
   reports the half-life range in which the 10-trade minimum is reachable.
7. Production code is not changed in this PR whatever the result.

## 6. Reproduction

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-lock.txt && pip install -e . --no-deps
python -m research.power.run_power_study --full --workers 4   # writes results/power/*
python -m pytest -q tests/test_power_study.py                 # includes the < 30 s smoke run
```

## Amendments

None at freeze.
