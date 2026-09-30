# Positioning: AATS family (canonical public vs legacy twins)

This note maps the four Advanced Algorithmic Trading Simulator (AATS) name
variants so portfolio viewers and future maintainers do **not** treat abandoned
private scaffolds as the active product surface.

## One-line roles

| Repo | Visibility | Role |
|------|------------|------|
| `jmiaie/Advanced_Algorithmic_Trading_Simulator_public` | **public** | **Canonical portfolio build** — statistical arbitrage & execution research engine; walk-forward discipline; publication pack. Recruiting / courseware surface. |
| `jmiaie/Advanced_Algorithmic_Trading_Simulator_private` | private | **Legacy placeholder** — thin README-only stub (Jan 2026). Retire when this public surface is confirmed as the sole AATS name. |
| `jmiaie/Advanced_Algorithmic_Trading_Simulator` | private | **Legacy scaffold** — early event-driven backtester notes; superseded by the public research engine. Retire / archive after owner confirmation. |
| `jmiaie/aats` | private | **Legacy short-name twin** — early upload scaffold; same intent as the long private names. Retire / archive after owner confirmation. |

Private twins are linked **by repository name only** here (no private URLs required
for portfolio readers). Collaborators who already have access know where they live.

## Recommendation

- **Use this public repo** for all new AATS / stat-arb portfolio claims, demos, and
  documentation.
- **Do not** dual-maintain features across the three private name variants.
- When open diffs on this public tree are merged and Jeff confirms, **archive or
  delete** the three private twins to cut inventory noise (they have no active code
  path beyond scaffolds / placeholders).

## Adjacent (not AATS-named)

| Repo | Note |
|------|------|
| `jmiaie/Statistical_Arbitrage_and_Conintegration_Strategic_Analyst` | Older public overlap. Consolidation policy: keep future verified public claims **here**; see [repository-consolidation.md](repository-consolidation.md). |
| `jmiaie/quant-research-portfolio-public` | Broader quant portfolio hub that may cite this study; not an AATS twin. |

## What belongs where

| Change type | Land in |
|-------------|---------|
| Portfolio README, positioning, quickstart, cheap test/docs fixes | **Advanced_Algorithmic_Trading_Simulator_public** |
| Research engine, walk-forward studies, publication pack | **this public repo** |
| New private-only experiments (if any) | Prefer a **named** private research hub — not the abandoned AATS name stubs |
| Portfolio copy that implies a private AATS twin is the product | Rewrite — funnel to this public repo |

## Explicit non-goals for this advance

- Do **not** fabricate Sharpe, return, drawdown, or turnover figures for the 2025
  walk-forward: zero FDR survivors → zero trades → no performance claim exists.
- Do **not** revive the private AATS scaffolds as parallel products.
- Do **not** merge owner-gated draft PRs as part of hygiene work.
