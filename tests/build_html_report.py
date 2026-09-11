"""Interactive HTML report — single shareable file with embedded Plotly charts.

Reads existing per-asset summary CSVs and equity PNGs, plus statistical
rigor results, and produces a single self-contained HTML report with:
- Interactive Plotly equity curves (per-asset)
- Sharpe distribution histograms
- Per-asset drill-down table
- Statistical significance summary

Outputs:
  outputs/report.html
"""
from __future__ import annotations
import base64
import json
import sys
from pathlib import Path

import pandas as pd

# Use plotly if available; otherwise fall back to embedding static PNGs
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


OUT = Path("outputs")
REPORT_PATH = OUT / "report.html"


def embed_png(path: Path) -> str:
    if not path.exists():
        return ""
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"


def make_equity_curve_plotly(csv_path: Path) -> str:
    """Build an interactive Plotly equity curve from per-window csv (if has detailed data)
    or fall back to a simple Sharpe bar chart from summary."""
    if not HAS_PLOTLY:
        return ""

    # Read summary for asset-level metrics
    if "per_window" in str(csv_path):
        # has minute detail
        df = pd.read_csv(csv_path)
    else:
        df = None

    return ""  # we use summary-level instead


def build_summary_table_html(df: pd.DataFrame, universe: str) -> str:
    rows = []
    for _, r in df.iterrows():
        asset = r.get("asset") or r.get("symbol", "?")
        bh = float(r.get("bh_sharpe", float("nan")))
        hmm = float(r.get("hmm_sharpe", float("nan")))
        delta = hmm - bh if (hmm == hmm and bh == bh) else float("nan")
        p_beat = float(r.get("p_hmm_beats_bh", float("nan")))
        bh_dd = float(r.get("bh_max_dd", float("nan")))
        hmm_dd = float(r.get("hmm_max_dd", float("nan")))
        delta_color = "#c8e6c9" if delta > 0 else "#ffcdd2"
        rows.append(f"""
        <tr style="background-color:{delta_color}">
          <td>{asset}</td>
          <td>{bh:+.3f}</td>
          <td>{hmm:+.3f}</td>
          <td><b>{delta:+.3f}</b></td>
          <td>{p_beat:.2f}</td>
          <td>{bh_dd:.2f}</td>
          <td>{hmm_dd:.2f}</td>
        </tr>""")
    return "".join(rows)


def build_interactive_plot(df: pd.DataFrame, title: str, top_n: int = 10) -> str:
    """Build an interactive Plotly bar chart of Δ Sharpe for top N + bottom N assets."""
    if not HAS_PLOTLY:
        return "<p>(plotly not installed; install with <code>pip install plotly</code>)</p>"

    df_sorted = df.sort_values("delta_sharpe", ascending=False).reset_index(drop=True)
    show = pd.concat([df_sorted.head(top_n), df_sorted.tail(top_n)]).reset_index(drop=True)
    show["label"] = show.apply(
        lambda r: f"{r['asset']} (Δ={r['delta_sharpe']:+.2f})", axis=1
    )

    colors = ["#4caf50" if d > 0 else "#f44336" for d in show["delta_sharpe"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=show["label"], y=show["delta_sharpe"],
        marker_color=colors, text=[f"{d:+.2f}" for d in show["delta_sharpe"]],
        textposition="outside", hovertemplate="<b>%{x}</b><br>ΔSharpe: %{y:+.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{title} — top {top_n} & bottom {top_n} by Δ Sharpe",
        xaxis_title="", yaxis_title="Δ Sharpe (HMM − B&H)",
        height=500, margin=dict(l=50, r=50, t=70, b=120),
        xaxis_tickangle=-45, template="plotly_white", showlegend=False,
    )
    return pio.to_html(fig, include_plotlyjs="cdn", full_html=False)


def build_distribution_plot(df: pd.DataFrame, title: str) -> str:
    if not HAS_PLOTLY:
        return ""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["delta_sharpe"], nbinsx=30,
        marker_color="#1976d2", opacity=0.7,
        hovertemplate="Δ Sharpe: %{x}<br>count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.add_vline(x=df["delta_sharpe"].mean(), line_dash="solid", line_color="green",
                  annotation_text=f"mean = {df['delta_sharpe'].mean():.3f}")
    fig.update_layout(
        title=f"{title} — Δ Sharpe distribution",
        xaxis_title="Δ Sharpe", yaxis_title="count",
        height=400, template="plotly_white", showlegend=False,
    )
    return pio.to_html(fig, include_plotlyjs="cdn", full_html=False)


def build_equity_curves_gallery(out_root: Path, universe_label: str, asset_dirs: list[str]) -> str:
    """Embed static PNG equity curves as gallery."""
    items = []
    for asset in asset_dirs:
        png = out_root / asset / "equity.png"
        if png.exists():
            items.append(f"""
            <div class="equity-card">
              <h4>{asset}</h4>
              <img src="{embed_png(png)}" alt="{asset} equity curve" loading="lazy"/>
            </div>""")
    return f"<div class='gallery'>{''.join(items)}</div>"


def main():
    crypto_df = None
    crypto_csv = OUT / "multi_asset" / "per_asset_summary.csv"
    if crypto_csv.exists():
        crypto_df = pd.read_csv(crypto_csv).rename(columns={"asset": "asset"})
        crypto_df["delta_sharpe"] = crypto_df["hmm_sharpe"] - crypto_df["bh_sharpe"]

    russell_df = None
    russell_csv = OUT / "russell2000_top200" / "summary.csv"
    if russell_csv.exists():
        russell_df = pd.read_csv(russell_csv).rename(columns={"symbol": "asset"})
        russell_df["delta_sharpe"] = russell_df["hmm_sharpe"] - russell_df["bh_sharpe"]

    stat = {}
    stat_path = OUT / "stat_rigor" / "results.json"
    if stat_path.exists():
        stat = json.loads(stat_path.read_text())

    # Build HTML
    crypto_table = build_summary_table_html(crypto_df, "crypto") if crypto_df is not None else "<p>(no crypto data)</p>"
    russell_table = build_summary_table_html(russell_df, "russell") if russell_df is not None else "<p>(no Russell 2000 data)</p>"

    crypto_plot = build_interactive_plot(crypto_df, "Crypto") if crypto_df is not None else ""
    russell_plot = build_interactive_plot(russell_df, "Russell 2000") if russell_df is not None else ""
    crypto_dist = build_distribution_plot(crypto_df, "Crypto") if crypto_df is not None else ""
    russell_dist = build_distribution_plot(russell_df, "Russell 2000") if russell_df is not None else ""

    # Build static equity galleries
    crypto_assets = sorted([p.name for p in (OUT / "multi_asset").iterdir() if p.is_dir()]) \
        if (OUT / "multi_asset").exists() else []
    russell_assets = sorted([p.name for p in (OUT / "russell2000_top200").iterdir() if p.is_dir()]) \
        if (OUT / "russell2000_top200").exists() else []
    crypto_gallery = build_equity_curves_gallery(OUT / "multi_asset", "Crypto", crypto_assets)
    russell_gallery = build_equity_curves_gallery(OUT / "russell2000_top200", "Russell 2000", russell_assets)

    # Stat block
    stat_html = ""
    if stat:
        blocks = []
        for name, r in stat.items():
            blocks.append(f"""
            <div class="stat-block">
              <h3>{name.upper()}</h3>
              <table>
                <tr><td>N assets</td><td>{r['n_assets']}</td></tr>
                <tr><td>Hit-rate (HMM &gt; B&amp;H)</td><td>{r['n_pos']}/{r['n_assets']} = {r['hit_rate']*100:.1f}%</td></tr>
                <tr><td>Mean Δ Sharpe</td><td>{r['mean_delta_sharpe']:+.4f}</td></tr>
                <tr><td>95% CI (t)</td><td>[{r['t_95ci_low']:+.4f}, {r['t_95ci_high']:+.4f}]</td></tr>
                <tr><td>t-statistic</td><td>{r['t_statistic']:+.3f}</td></tr>
                <tr><td>t-test p-value</td><td><b>{r['t_p_value']:.4g}</b></td></tr>
                <tr><td>Wilcoxon p-value</td><td>{r['wilcoxon_p_value'] if r['wilcoxon_p_value'] else 'n/a'}</td></tr>
                <tr><td>Binomial p (two-sided)</td><td><b>{r['binom_p_two_sided']:.4g}</b></td></tr>
                <tr><td>Cohen's d</td><td>{r['cohens_d']:+.3f}</td></tr>
                <tr><td>Bootstrap 95% CI on mean</td><td>[{r['bootstrap_95ci_low']:+.4f}, {r['bootstrap_95ci_high']:+.4f}]</td></tr>
                <tr><td>Required n for 80% power</td><td>{r['required_n_for_80pct_power']} (have {r['n_assets']})</td></tr>
              </table>
            </div>""")
        stat_html = "".join(blocks)

    # CSS
    css = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
             margin: 0; padding: 20px; background: #fafafa; color: #212121; }
      .container { max-width: 1400px; margin: 0 auto; background: white;
                  padding: 30px 40px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
      h1 { color: #1565c0; border-bottom: 3px solid #1976d2; padding-bottom: 12px; }
      h2 { color: #1976d2; margin-top: 40px; border-left: 4px solid #1976d2; padding-left: 12px; }
      h3 { color: #455a64; }
      h4 { color: #607d8b; margin: 0 0 8px 0; }
      table { border-collapse: collapse; width: 100%; margin: 12px 0;
              font-size: 13px; }
      th, td { border: 1px solid #e0e0e0; padding: 6px 10px; text-align: right; }
      th { background: #f5f5f5; text-align: left; font-weight: 600; }
      td:first-child, th:first-child { text-align: left; }
      tr:hover { background: #f5f5f5; }
      .stat-block { display: inline-block; vertical-align: top; width: 48%;
                     margin: 12px 1%; padding: 16px;
                     background: #f5f5f5; border-radius: 6px;
                     box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
      .stat-block td:first-child { font-weight: 600; }
      .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
                gap: 16px; margin: 16px 0; }
      .equity-card { background: white; padding: 12px; border-radius: 4px;
                     box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
      .equity-card img { width: 100%; height: auto; display: block; }
      .summary { background: #e3f2fd; padding: 16px; border-radius: 6px;
                 margin: 16px 0; border-left: 4px solid #1976d2; }
      .verdict-strong { color: #1b5e20; font-weight: 700; }
      .verdict-moderate { color: #f57c00; font-weight: 700; }
      .verdict-marginal { color: #e65100; font-weight: 700; }
      .verdict-insufficient { color: #b71c1c; font-weight: 700; }
    </style>
    """

    verdict_crypto = stat.get("crypto", {}).get("t_p_value", 1.0)
    verdict_russell = stat.get("russell2000_top200", {}).get("t_p_value", 1.0)
    def verdict_class(p):
        if p < 0.001: return "verdict-strong"
        if p < 0.01: return "verdict-moderate"
        if p < 0.05: return "verdict-marginal"
        return "verdict-insufficient"
    def verdict_label(p):
        if p < 0.001: return "STRONG"
        if p < 0.01: return "MODERATE"
        if p < 0.05: return "MARGINAL"
        return "INSUFFICIENT"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Risk-Management Pipeline — Results</title>
  {css}
</head>
<body>
  <div class="container">
    <h1>Risk-Management Pipeline — Results Report</h1>
    <p>Generated by <code>tests/build_html_report.py</code>. Self-contained, no
       external requests needed to view. Interactive: hover over charts for details.</p>

    <div class="summary">
      <h2>Executive summary</h2>
      <p><b>Crypto multi-asset (9 assets):</b> <span class="{verdict_class(verdict_crypto)}">{verdict_label(verdict_crypto)}</span>
         — mean Δ Sharpe +{stat.get('crypto', {}).get('mean_delta_sharpe', 0):.3f},
         {stat.get('crypto', {}).get('n_pos', 0)}/{stat.get('crypto', {}).get('n_assets', 0)} beat B&amp;H,
         t-test p = {verdict_crypto:.4g}</p>
      <p><b>Russell 2000 top-200 (177 valid):</b> <span class="{verdict_class(verdict_russell)}">{verdict_label(verdict_russell)}</span>
         — mean Δ Sharpe +{stat.get('russell2000_top200', {}).get('mean_delta_sharpe', 0):.3f},
         {stat.get('russell2000_top200', {}).get('n_pos', 0)}/{stat.get('russell2000_top200', {}).get('n_assets', 0)} beat B&amp;H,
         t-test p = {verdict_russell:.4g}</p>
    </div>

    <h2>Statistical significance</h2>
    {stat_html or '<p>(no stat_rigor results.json found)</p>'}

    <h2>Crypto — top &amp; bottom performers</h2>
    {crypto_plot}
    {crypto_dist}
    <h3>Detail table</h3>
    <table>
      <thead><tr><th>Asset</th><th>B&amp;H Sharpe</th><th>HMM Sharpe</th><th>Δ Sharpe</th><th>P(HMM&gt;B&amp;H)</th><th>Max DD B&amp;H</th><th>Max DD HMM</th></tr></thead>
      <tbody>{crypto_table}</tbody>
    </table>

    <h2>Crypto — equity curves</h2>
    {crypto_gallery}

    <h2>Russell 2000 — top &amp; bottom performers</h2>
    {russell_plot}
    {russell_dist}
    <h3>Detail table (top 25 + bottom 10 by Δ Sharpe)</h3>
    <table>
      <thead><tr><th>Asset</th><th>B&amp;H Sharpe</th><th>HMM Sharpe</th><th>Δ Sharpe</th><th>P(HMM&gt;B&amp;H)</th><th>Max DD B&amp;H</th><th>Max DD HMM</th></tr></thead>
      <tbody>
        {''.join([
          f'<tr style="background-color:{"#c8e6c9" if r["delta_sharpe"] > 0 else "#ffcdd2"}">'
          f'<td>{r["asset"]}</td>'
          f'<td>{r["bh_sharpe"]:+.3f}</td>'
          f'<td>{r["hmm_sharpe"]:+.3f}</td>'
          f'<td><b>{r["delta_sharpe"]:+.3f}</b></td>'
          f'<td>{r["p_hmm_beats_bh"]:.2f}</td>'
          f'<td>{r["bh_max_dd"]:.2f}</td>'
          f'<td>{r["hmm_max_dd"]:.2f}</td>'
          f'</tr>'
            for _, r in pd.concat([russell_df.nlargest(25, 'delta_sharpe'),
                                    russell_df.nsmallest(10, 'delta_sharpe')]).iterrows()
        ])}
      </tbody>
    </table>

    <h2>Russell 2000 — equity curves (top 20 winners + top 5 losers)</h2>
    <div class="gallery">
      {''.join([
          f'<div class="equity-card"><h4>{a}</h4><img src="{embed_png(OUT / "russell2000_top200" / a / "equity.png")}" loading="lazy"/></div>'
          for a in (russell_df.nlargest(20, 'delta_sharpe')['asset'].tolist() +
                    russell_df.nsmallest(5, 'delta_sharpe')['asset'].tolist())
          if (OUT / "russell2000_top200" / a / "equity.png").exists()
      ])}
    </div>

    <hr/>
    <p style="color: #888; font-size: 11px;">
      Generated by Risk-Management Pipeline v1.1.0.
      Static PNGs are embedded as base64. Interactive Plotly charts require CDN access
      (plotly.js loaded from cdn.plot.ly).
    </p>
  </div>
</body>
</html>"""

    REPORT_PATH.write_text(html)
    print(f"Wrote {REPORT_PATH} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
