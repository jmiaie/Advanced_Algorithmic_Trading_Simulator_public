# Known defects and previously unreported results (2026-11-14)

This note discloses two code defects and one reporting gap found in a review of the accepted
`statarb_hist_etf_wf_v4` study. The frozen artifacts, configuration and publication pack are
**unchanged**, and nothing was re-run. See also
[`post-publication-bh-correction.md`](post-publication-bh-correction.md).

**The 2025 result is not affected by anything here.** The three 2025 windows had zero screen
survivors, so no position was sized, traded or marked.

## 1. Position sizing does not trade the tested spread

`src/stat_arb_engine/wf_v3_sizing.py` splits the gross dollar allocation between the two legs in the
ratio 1 : β. But β is the slope of a regression on **price levels**, so it is a ratio of shares
(units of y per unit of x), not of dollars. A hedge that matches the tested spread
`y − β·x` holds β shares of x per share of y, which is a dollar ratio of `price_y : β·price_x`.

Effect: in the development windows that traded, the position was not the spread the cointegration
test examined, and it was not market-neutral. In the last traded window (XLU/XLV) the static-OLS
position averaged about 100k net long on 174k gross (about 58%).

The code is left as it is, because changing it would change the behaviour behind frozen artifacts.
Any future study must size by shares (or re-estimate β on dollar-neutral terms) first.

## 2. No end-of-window liquidation cost

`src/stat_arb_engine/wf_v4_backtest.py` marks any position still open at the end of a test window
to market and charges no exit cost. Returns for windows that ended with an open position are
slightly overstated.

## 3. Development-window out-of-sample results that the write-up did not report

The technical paper says the screen "fires when opportunities exist in-sample (4/1,724)". The four
development windows in which a pair was admitted were also traded out of sample, and those results
are in the committed artifact `results/historical_oos/statarb_hist_etf_wf_v4_dev_formation.json`.
They were not reported in the paper. Net return on allocated capital, base cost scenario:

| Window (test period) | Pair | Static OLS | Trades | Kalman | Trades |
|---|---|---|---|---|---|
| 2017-10-03 → 2018-01-02 | XLP/XLU | −3.76% | 2 | −1.04% | 7 |
| 2018-01-03 → 2018-04-04 | XLP/XLU | −3.20% | 2 | +2.75% | 4 |
| 2018-04-05 → 2018-07-03 | XLP/XLU | +2.30% | 3 | −1.33% | 2 |
| 2018-07-05 → 2018-10-02 | XLU/XLV | +1.54% | 2 | +0.99% | 2 |

How to read this:

- These numbers carry both defects above, so they are not a clean test of the strategy.
- Four overlapping windows, three of them the same pair, with 2–7 trades each, are far too few to
  estimate performance. The sum across windows is negative for static OLS and slightly positive
  for Kalman; neither is distinguishable from zero.
- They do not support a claim that admitted pairs were profitable. "The screen fires when
  opportunities exist" should be read only as "the screen admitted a pair in four windows".

## 4. Wording: what "FDR survivors" counts

A pair is counted as a survivor only if it is rejected by Benjamini-Hochberg **and** passes the
residual ADF test (`src/stat_arb_engine/wf_v3_pairs.py`, `eligible = fdr_rejected & adf_passes`).
The paper describes ADF as a secondary diagnostic that "does not by itself admit a pair". That is
true, but ADF can **veto** a pair, so the survivor counts are counts of pairs passing both tests.

## 5. Other limits of the pair screen

- Each pair's history is cut to the dates on which both tickers have prices, and a pair whose test
  raises an error is skipped. Both reduce the number of hypotheses the FDR control sees (46 in the
  earliest windows, 55 later).
- Power: a separate calibration (open pull request "power / null-calibration study for the stat-arb
  screen") shows the screen reliably detects only fast mean reversion (half-life of about a week).
