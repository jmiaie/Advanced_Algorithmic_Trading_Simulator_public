# Positioning: AATS family (canonical public vs legacy twins)

This note maps the four Advanced Algorithmic Trading Simulator (AATS) name
variants so portfolio viewers and future maintainers do **not** treat abandoned
private scaffolds as the active product surface.

## One-line roles

| Repo | Visibility | Role |
|------|------------|------|
| `jmiaie/Advanced_Algorithmic_Trading_Simulator_public` | **public** | **Canonical portfolio build** — statistical arbitrage & execution research engine; walk-forward discipline; publication pack. Recruiting / courseware surface. |
| Three non-public name variants | private | **Legacy placeholders / scaffolds** — README-only stubs and early backtester notes, superseded by this public research engine. Slated for retirement. |

The non-public variants are intentionally not named or linked here.

## Recommendation

- **Use this public repo** for all new AATS / stat-arb portfolio claims, demos, and
  documentation.
- **Do not** dual-maintain features across the non-public name variants.

## Adjacent (not AATS-named)

| Repo | Note |
|------|------|
| `jmiaie/Statistical_Arbitrage_and_Conintegration_Strategic_Analyst` | Older, overlapping exploratory repository; superseded by this one. |
| `jmiaie/quant-research-portfolio-public` | Broader quant portfolio hub that may cite this study; not an AATS twin. |

## Explicit non-goals

- Do **not** fabricate Sharpe, return, drawdown, or turnover figures for the 2025
  walk-forward: zero FDR survivors → zero trades → no performance claim exists.
- Do **not** revive the private AATS scaffolds as parallel products.
