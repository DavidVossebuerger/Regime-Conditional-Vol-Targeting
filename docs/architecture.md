# Pipeline Architecture

End-to-end flow of the HMM wc-feat risk-management system. Each component has
a single responsibility; all data is shifted so no lookahead leaks.

## End-to-end flow

```mermaid
flowchart TB
    %% --- Inputs ---
    subgraph INPUT["Input (daily/hourly OHLC bars)"]
        direction LR
        CRYPTO["Crypto: data/crypto_BTC_USD_1m*.parquet<br/>data/*_usd_1h.csv"]
        LSE["Equities: LSE API<br/>src/data_io/lse_loader.py<br/>(cached in data/cache/)"]
    end

    %% --- Feature engineering (lagged, no future info) ---
    subgraph FEAT["Feature engineering (per asset)"]
        direction LR
        F1["log_return = log(p_t / p_{t-1})"]
        F2["rv = rolling(24h or 20d) std × √periods_per_year"]
        F3["lags 1..5 of ret and rv"]
        F4["rolling mean / std / skew / kurt"]
        F5["target = rv.shift(-1)<br/>(next-period RV — the prediction target)"]
        F1 --> F2 --> F3 --> F4 --> F5
    end

    INPUT --> FEAT

    %% --- Vol forecast (RF) ---
    subgraph RF["Random Forest vol forecast (per asset, fit ONCE)"]
        RF1["Train on first 80% of bars"]
        RF2["Features in, next-bar RV out"]
        RF3["Predict on test fold → pred_vol(t+1)"]
        RF1 --> RF2 --> RF3
    end

    F5 --> RF1

    %% --- Worst-case feature (V_wc, optional) ---
    subgraph WC["Worst-case feature (Feng 2019, optional 4th HMM input)"]
        direction LR
        WC1["rolling 168h log_returns"]
        WC2["V_wc(η) = η⁻¹ · log(mean(exp(η · r)))<br/>= worst-case expected log return"]
        WC3["η picked by calibration snippet<br/>(smallest η where |V_wc - V|/|V| ≥ 50%)"]
        WC1 --> WC2 --> WC3
    end

    F5 --> WC1

    %% --- HMM regime classifier ---
    subgraph HMM["Gaussian HMM regime classifier (K=3, per WF window)"]
        direction LR
        HMM_IN["Features (all lagged):<br/>rv_zscore_168h, vol_of_vol_72h,<br/>vol_return, ±V_wc"]
        HMM_FIT["Fit on rolling 180d training slice<br/>(GaussianHMM, log-domain forward)"]
        HMM_OUT["P(state=k | features_t) for k ∈ {0,1,2}"]
        HMM_IN --> HMM_FIT --> HMM_OUT
    end

    RF3 --> HMM_IN
    WC3 --> HMM_IN

    %% --- Per-state target (offline, walk-forward) ---
    subgraph TARGET["Per-state target (brute-force, per WF window)"]
        direction LR
        T1["For each combo (target₁, target₂, target₃):"]
        T2["Build mix position series:<br/>Σ_k P_k(t) · clip(target_k / pred_vol, 0, 1)"]
        T3["Compute training Sharpe<br/>(subtract TC)"]
        T4["Pick combo with best Sharpe"]
        T1 --> T2 --> T3 --> T4
    end

    HMM_OUT --> T1
    RF3 --> T1

    %% --- Live position (test fold) ---
    subgraph LIVE["Live position sizing (per bar)"]
        direction LR
        L1["For each hour t:<br/>pos_k(t) = clip(target_k / pred_vol(t+1), 0, 1)"]
        L2["pos(t) = Σ_k P_k(t) · pos_k(t)<br/>(convex mixture)"]
        L3["TC deducted on |Δpos(t)| · TC_per_side"]
        L1 --> L2 --> L3
    end

    T4 --> L1
    RF3 --> L1
    HMM_OUT --> L1

    %% --- Outputs ---
    subgraph OUT["Outputs"]
        O1["equity.png — equity / vol / DD"]
        O2["summary.json — sharpe, max_dd, pct_within"]
        O3["per_window.csv — OOS metrics per walk-forward slice"]
        O4["yearly.csv — regime breakdown"]
    end

    L3 --> O1
    L3 --> O2
    L3 --> O3
    L3 --> O4
```

## Component responsibilities

| Component | Output | Fit cadence | What it learns |
|---|---|---|---|
| **Random Forest** | `pred_vol(t+1)` | once per asset | vol magnitude from lagged features |
| **Gaussian HMM** | `P(state=k | features_t)` | once per walk-forward window | regime probabilities (calm / neutral / stress) |
| **Per-state target** | `target_k ∈ {0.2..1.0}` | once per walk-forward window | risk-appetite per regime (joint brute-force over 11³ = 1331 combos) |
| **Worst-case feature** | `V_wc(t)` | recomputed each test bar | tail-risk magnitude from rolling 168h window |

## Key invariants

1. **No lookahead.** All features use `shift(+lag)` or rolling windows ending at time `t`.
   Only the *target* uses `shift(-1)`.
2. **Walk-forward honest.** Each test slice sees only HMMs/targets trained on data
   strictly before that slice.
3. **Convex mixture.** `Σ_k P_k(t) = 1` and `pos ∈ [0, 1]`, so position is bounded.
4. **Calibration snippet is in train fold only.** η is picked from a random 90-day
   slice of the train fold — never sees test data.
5. **Block bootstrap on 60-day blocks** (1 week for daily bars, 1 week for hourly).

## Where time is spent

```
Per asset, per run:
  LSE fetch (cold):     ~1s
  LSE fetch (cached):   ~0s
  RF fit (80% × 17 features):      ~0.3s
  HMM × N windows (K=3, log-domain): ~0.3s × N
  Per-state brute force (11³):     ~0.5s
  Bootstrap 500 × 60-day blocks:   ~0.5s
  ──────────────────────────────────
  Total per asset:     ~1-3s cold, ~0.5-1s warm
  200 assets:           ~5-15 min total
```

## Without HMM (static-soft fallback)

```mermaid
flowchart LR
    RF["RF vol forecast"] --> POS["pos(t) = clip(target / pred_vol(t+1), 0, 1)<br/>(single target for all time)"]
    RF -.no HMM, no WC.-> POS
```

The static-soft config uses **one** `target_vol` for the entire history (picked
by in-sample Sharpe maximization on the calibration snippet). Used as the
`static_soft` baseline in every comparison; HMM wc-feat consistently beats it.
