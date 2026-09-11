# Anchored Walk-Forward Test — Strictest No-Lookahead Validation

Squiggle (Discord) implicitly asked: "is the rolling WF inflating the edge by
re-fitting the HMM on data that overlaps with later test slices?"

To answer this rigorously, we ran the **anchored WF** test on all 1039
equities_yf assets:
- **HMM** is fit ONCE on the first WF window's train slice (~504 bars)
- **HMM is then frozen** for all subsequent windows — no re-fit
- **Per-state targets** are also computed ONCE on the first window and reused

This is the strictest possible no-lookahead test: the regime detector never
sees any data after its initial training window.

## Headline result — Anchored is **slightly better** than Rolling

| Mode | N | Hit rate | Mean ΔSharpe | t-stat | Wilcoxon p | Cohen's d |
|---|---|---|---|---|---|---|
| **Rolling** (HMM re-fit per window) | 1039 | 76.6 % | +0.129 | +18.0 | 4.3 × 10⁻⁷⁹ | +0.560 |
| **Anchored** (HMM frozen) | 1039 | 74.1 % | **+0.160** | +15.8 | 1.4 × 10⁻⁶⁵ | +0.491 |
| **Δ (Rolling − Anchored)** | — | — | **−0.031** | **−3.20** | **p = 0.001** | — |

- **Anchored beats Rolling** by ΔSharpe +0.031 on average (paired t = −3.20,
  p = 0.001).
- **Anchored alone** is still wildly significant: Wilcoxon p = **1.4 × 10⁻⁶⁵** on
  1039 paired observations.
- The "lookahead concern" was unfounded — re-fitting the HMM per window
  actually **dilutes** the edge, not amplifies it.

## Why this is reassuring

1. **No data leakage.** The frozen HMM was fit only on data from the first
   walk-forward window. It never saw any bar that it was later evaluated on.
2. **Edge is robust to HMM timing.** Whether we re-fit the HMM every 6 months
   or freeze it for 10+ years, the strategy produces a positive edge. The
   mechanism is therefore **not** "HMM cleverly detects regime shifts in
   real time" — it's something more fundamental.
3. **Cohen's d ≈ 0.5** even with the strictest test → a real medium-to-large
   effect size, not an artifact of how the HMM is fit.

## By quartile (B&H Sharpe)

| Quartile | N | Rolling hit rate | Anchored hit rate |
|---|---|---|---|
| Q1 (worst B&H) | 260 | 72.3 % | 70.4 % |
| Q2 | 260 | 80.0 % | 77.3 % |
| Q3 | 259 | 81.5 % | 79.2 % |
| Q4 (best B&H) | 260 | 72.7 % | 69.6 % |

Both modes follow a similar pattern: the edge is somewhat smaller for the
worst and best quartiles of B&H performance, and largest in the middle.

## Top movers (asset-level)

### 10 largest improvements from rolling → anchored

| Symbol | B&H Sharpe | Δ Rolling | Δ Anchored | Δ improvement |
|---|---|---|---|---|
| MP (Matador Resources) | −0.36 | +1.57 | **+3.39** | **+1.82** |
| NE | +0.14 | −0.08 | +1.66 | +1.74 |
| ZETA | +1.68 | −0.33 | +1.34 | +1.67 |
| APP | −0.53 | +0.10 | +1.56 | +1.45 |
| SITM | +0.78 | +0.36 | +1.74 | +1.39 |
| TOST (Toast) | +0.92 | −0.20 | +1.11 | +1.32 |
| XMR-USD | +0.66 | +0.40 | +1.68 | +1.28 |
| AXTI | +0.67 | +0.08 | +1.25 | +1.17 |
| GAP | +0.11 | +0.08 | +1.23 | +1.15 |
| WFRD | −0.92 | +0.95 | +2.07 | +1.12 |

### 10 largest regressions (rolling was better)

| Symbol | B&H Sharpe | Δ Rolling | Δ Anchored | Δ loss |
|---|---|---|---|---|
| PCVX (Vaxcyte) | +0.95 | +1.15 | −0.93 | **−2.08** |
| HIMS (Hims & Hers) | −0.41 | +0.41 | −1.24 | −1.65 |
| KD (Kyndryl) | +0.10 | +1.71 | +0.24 | −1.47 |
| YETI | +0.42 | +0.67 | −0.73 | −1.40 |
| TTMI | +1.19 | +0.53 | −0.51 | −1.04 |
| GNRC | +0.28 | +0.54 | −0.44 | −0.98 |
| BOOT | +0.69 | +0.07 | −0.79 | −0.86 |
| BROS | −0.08 | +0.67 | −0.18 | −0.85 |
| FTM-USD | −0.03 | +0.17 | −0.67 | −0.84 |
| P | +0.66 | +0.63 | −0.21 | −0.83 |

The regressions are real — for some assets the rolling HMM was clearly
helping. But on the population level, anchoring wins.

## Interpretation

The most likely interpretation: **the HMM is not the alpha source**. The
alpha comes from the **per-state vol-targeting rule** + **defensive position
sizing in stress regimes**. The HMM is just a useful label that picks up
roughly similar regimes across assets. Whether it's re-fit or frozen, the
strategy produces a positive edge because the underlying sizing rule is
sensible.

Practically, this means:
- A **production deployment can use a frozen HMM** trained at system start.
- **Re-fit cadence matters less than expected** — even a frozen HMM works.
- The edge is **portable across assets** without per-asset regime tuning.

## How to reproduce

```bash
# Rolling (default)
python src/equities_runner.py --data-source yf \
    --output-dir outputs/equities_yf --tc-per-side 0.0002

# Anchored (frozen HMM)
python src/equities_runner.py --data-source yf \
    --output-dir outputs/equities_yf_anchored --tc-per-side 0.0002 \
    --anchored-wf

# Compare
python - <<'PY'
import pandas as pd
import numpy as np
from scipy import stats

r = pd.read_csv("outputs/equities_yf/summary.csv").rename(
    columns={"sharpe_delta": "delta_rolling"})
a = pd.read_csv("outputs/equities_yf_anchored/summary.csv").rename(
    columns={"sharpe_delta": "delta_anchored"})
m = r.merge(a[["symbol", "delta_anchored"]], on="symbol").dropna()
t, p = stats.ttest_rel(m["delta_rolling"], m["delta_anchored"])
print(f"paired t (rolling - anchored) = {t:+.2f}, p = {p:.3g}")
print(f"mean rolling   = {m.delta_rolling.mean():+.4f}")
print(f"mean anchored  = {m.delta_anchored.mean():+.4f}")
PY
```

## Caveats

1. **First-window dependence.** The anchored HMM is fit on the first WF
   window's train slice. If that window happens to fall on an unusual
   market regime (e.g., right before a crash), the HMM calibration could
   be poor. The 1039-asset average absorbs this idiosyncratic risk but a
   specific deployment might still want a warm-up period.
2. **Equity-only.** This test was only run on equities_yf (1039 assets).
   We have not yet re-run crypto with anchored WF. Given crypto's wider
   regime shifts (more obvious bull/bear cycles), the anchored vs rolling
   comparison could differ there.
3. **Two-mode comparison, not exhaustive.** We tested only "rolling" (re-fit
   per window) and "anchored" (frozen). A hybrid (e.g., re-fit every N
   windows) could give intermediate results. Not pursued.

## Conclusion

The edge is **not** an artifact of rolling-WF HMM adaptation. It survives
the strictest no-lookahead test (frozen HMM trained only on the first
window) with Wilcoxon p = 1.4 × 10⁻⁶⁵. This is the most defensible
evidence yet that the RCVT edge is real.
