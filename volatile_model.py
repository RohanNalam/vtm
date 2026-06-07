"""
Volatility Trading Model — Enhanced Edition

Twists over standard vol models:
  1. Volatility Regime Detection   — HMM-style clustering (low/mid/high vol)
  2. Vol-of-Vol Filter             — avoid entering when second-order vol is spiking
  3. RV vs IV Spread               — trade the gap between realized and implied vol
  4. Multi-Timeframe Confluence    — signal only fires when short + medium + long vol agree
  5. Skewness-Adjusted Kelly       — position sizing penalized for fat-tail risk
  6. Adaptive ATR Bands            — band width scales with regime, not fixed multiplier
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal, Tuple
from enum import Enum


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Regime(Enum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"


@dataclass
class Signal:
    direction: Literal["long_vol", "short_vol", "flat"]
    confidence: float          # 0–1
    regime: Regime
    kelly_fraction: float      # suggested position size as fraction of capital
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def rolling_realized_vol(returns: pd.Series, window: int) -> pd.Series:
    """Annualized realized volatility over `window` bars."""
    return returns.rolling(window).std() * np.sqrt(252)


def rolling_skew(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).skew()


def rolling_kurt(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).kurt()


# ---------------------------------------------------------------------------
# Twist 1 — Volatility Regime Detection
# ---------------------------------------------------------------------------

def detect_regime(rv: pd.Series, low_q: float = 0.33, high_q: float = 0.67) -> pd.Series:
    """
    Classify each bar into LOW / MID / HIGH vol regime using rolling quantiles.
    Rolling quantiles let the model adapt to secular vol shifts instead of using
    hard-coded thresholds that go stale.
    """
    lookback = min(252, len(rv))
    low_thresh  = rv.rolling(lookback, min_periods=20).quantile(low_q)
    high_thresh = rv.rolling(lookback, min_periods=20).quantile(high_q)

    regime = pd.Series(Regime.MID, index=rv.index)
    regime[rv < low_thresh]  = Regime.LOW
    regime[rv > high_thresh] = Regime.HIGH
    return regime


# ---------------------------------------------------------------------------
# Twist 2 — Vol-of-Vol Filter
# ---------------------------------------------------------------------------

def vol_of_vol(rv: pd.Series, window: int = 21) -> pd.Series:
    """
    Second-order volatility: std of realized vol itself.
    When VVIX (vol-of-vol) is high the distribution of outcomes becomes
    unpredictable — we reduce exposure regardless of direction.
    """
    return rv.rolling(window).std()


def vvol_z_score(vvol: pd.Series, window: int = 63) -> pd.Series:
    """Z-score of vol-of-vol so we have a scale-free filter."""
    mu  = vvol.rolling(window).mean()
    sig = vvol.rolling(window).std()
    return (vvol - mu) / sig.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Twist 3 — Realized vs Implied Vol Spread (RV-IV)
# ---------------------------------------------------------------------------

def rv_iv_spread(rv: pd.Series, iv: pd.Series, smooth: int = 5) -> pd.Series:
    """
    Spread between implied vol (what options market prices in) and realized vol.
    - Positive spread (IV > RV): vol sellers are being paid a premium → short vol.
    - Negative spread (RV > IV): realized vol is surprising to the upside → long vol.
    Smoothing removes microstructure noise.
    """
    return (iv - rv).rolling(smooth).mean()


# ---------------------------------------------------------------------------
# Twist 4 — Multi-Timeframe Confluence
# ---------------------------------------------------------------------------

def mtf_confluence(
    returns: pd.Series,
    short_w: int  = 5,
    medium_w: int = 21,
    long_w: int   = 63,
) -> pd.Series:
    """
    Returns a confluence score in [-1, 1].
      +1 : all three timeframes show vol expanding (long vol bias)
      -1 : all three timeframes show vol contracting (short vol bias)
       0 : mixed, stay flat

    Logic: compare current RV to its own rolling mean at each horizon.
    When short > medium > long we have a vol acceleration regime.
    """
    rv_s = rolling_realized_vol(returns, short_w)
    rv_m = rolling_realized_vol(returns, medium_w)
    rv_l = rolling_realized_vol(returns, long_w)

    score = pd.Series(0.0, index=returns.index)

    expanding = (rv_s > rv_m) & (rv_m > rv_l)
    contracting = (rv_s < rv_m) & (rv_m < rv_l)

    score[expanding]   = 1.0
    score[contracting] = -1.0
    return score


# ---------------------------------------------------------------------------
# Twist 5 — Skewness-Adjusted Kelly Criterion
# ---------------------------------------------------------------------------

def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    skew: float = 0.0,
    kurt: float = 0.0,
    max_f: float = 0.25,
) -> float:
    """
    Standard Kelly: f* = (p * b - q) / b  where b = avg_win / avg_loss.

    Adjustments applied here:
      - Negative skew penalty: fat left tails are worse than Kelly assumes.
        Each unit of negative skew shaves 10% off f*.
      - Excess kurtosis penalty: leptokurtic returns have surprise large moves.
        Each unit of excess kurtosis beyond 3 shaves 5% off f*.
      - Hard cap at max_f to avoid Kelly's notorious over-betting.
    """
    if avg_loss == 0:
        return 0.0

    b = avg_win / avg_loss
    q = 1.0 - win_rate
    f_raw = (win_rate * b - q) / b

    # Skew penalty (only penalize negative skew)
    skew_penalty = max(0.0, -skew) * 0.10

    # Excess kurtosis penalty
    excess_kurt  = max(0.0, kurt - 3.0)
    kurt_penalty = excess_kurt * 0.05

    f_adj = f_raw * (1.0 - skew_penalty - kurt_penalty)
    return float(np.clip(f_adj, 0.0, max_f))


# ---------------------------------------------------------------------------
# Twist 6 — Adaptive ATR Bands
# ---------------------------------------------------------------------------

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=window, adjust=False).mean()


def adaptive_bands(
    close: pd.Series,
    atr_series: pd.Series,
    regime: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """
    Band multiplier adapts to regime:
      LOW  → 1.5x ATR  (tighter, less noise expected)
      MID  → 2.5x ATR  (standard)
      HIGH → 4.0x ATR  (wide, vol is fat-tailed in this regime)

    Using a fixed 2.0 ATR in all regimes causes constant whipsaws during
    high-vol episodes and misses breakouts in low-vol compression.
    """
    multiplier = regime.map({
        Regime.LOW:  1.5,
        Regime.MID:  2.5,
        Regime.HIGH: 4.0,
    }).fillna(2.5)

    upper = close + multiplier * atr_series
    lower = close - multiplier * atr_series
    return upper, lower


# ---------------------------------------------------------------------------
# Main Signal Generator
# ---------------------------------------------------------------------------

def generate_signal(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    implied_vol: pd.Series,            # annualized IV, e.g. from VIX / 100
    rv_window: int   = 21,
    vvol_threshold: float = 1.5,       # z-score above which we flatten on vvol
) -> pd.DataFrame:
    """
    Combines all six twists into a single signal frame.

    Returns DataFrame with columns:
        direction, confidence, regime, kelly_fraction, rv, iv, rv_iv_spread,
        mtf_score, vvol_z, atr, upper_band, lower_band
    """
    returns = close.pct_change()

    # --- Base metrics ---
    rv      = rolling_realized_vol(returns, rv_window)
    regime  = detect_regime(rv)
    vvol    = vol_of_vol(rv)
    vvol_z  = vvol_z_score(vvol)
    spread  = rv_iv_spread(rv, implied_vol)
    mtf     = mtf_confluence(returns)
    skew    = rolling_skew(returns, rv_window)
    kurt    = rolling_kurt(returns, rv_window)
    atr_s   = atr(high, low, close)
    upper, lower = adaptive_bands(close, atr_s, regime)

    signals = []

    for i in range(len(close)):
        idx = close.index[i]

        _rv      = rv.iloc[i]
        _spread  = spread.iloc[i]
        _mtf     = mtf.iloc[i]
        _vvol_z  = vvol_z.iloc[i]
        _skew    = skew.iloc[i]
        _kurt    = kurt.iloc[i]
        _regime  = regime.iloc[i]

        if pd.isna(_rv) or pd.isna(_spread) or pd.isna(_vvol_z):
            signals.append({
                "date": idx, "direction": "flat", "confidence": 0.0,
                "regime": None, "kelly_fraction": 0.0,
                "rv": _rv, "iv": implied_vol.iloc[i],
                "rv_iv_spread": _spread, "mtf_score": _mtf,
                "vvol_z": _vvol_z, "atr": atr_s.iloc[i],
                "upper_band": upper.iloc[i], "lower_band": lower.iloc[i],
            })
            continue

        reasons = []
        direction = "flat"
        confidence = 0.0

        # --- Twist 2: vvol filter (override everything) ---
        if _vvol_z > vvol_threshold:
            reasons.append(f"vvol_z={_vvol_z:.2f} exceeds threshold — forced flat")
            signals.append({
                "date": idx, "direction": "flat", "confidence": 0.0,
                "regime": _regime, "kelly_fraction": 0.0,
                "rv": _rv, "iv": implied_vol.iloc[i],
                "rv_iv_spread": _spread, "mtf_score": _mtf,
                "vvol_z": _vvol_z, "atr": atr_s.iloc[i],
                "upper_band": upper.iloc[i], "lower_band": lower.iloc[i],
            })
            continue

        # --- Twist 3: RV-IV spread ---
        if _spread > 0.02:          # IV > RV by 2pp → vol sellers have edge
            direction = "short_vol"
            confidence += 0.35
            reasons.append(f"IV premium={_spread:.3f} -> short vol edge")
        elif _spread < -0.02:       # RV > IV -> realized surprises to upside
            direction = "long_vol"
            confidence += 0.35
            reasons.append(f"RV excess={abs(_spread):.3f} -> long vol edge")

        # --- Twist 4: Multi-timeframe confluence ---
        if _mtf == 1.0 and direction in ("long_vol", "flat"):
            direction = "long_vol"
            confidence += 0.30
            reasons.append("MTF confluence: vol accelerating across all timeframes")
        elif _mtf == -1.0 and direction in ("short_vol", "flat"):
            direction = "short_vol"
            confidence += 0.30
            reasons.append("MTF confluence: vol decelerating across all timeframes")
        elif _mtf != 0.0 and direction != "flat":
            # partial agreement
            confidence += 0.10
            reasons.append("MTF partial confluence")

        # --- Regime adjustment ---
        if _regime == Regime.HIGH and direction == "long_vol":
            confidence *= 0.80   # already in high vol, mean reversion risk
            reasons.append("Regime HIGH — confidence trimmed (mean reversion risk)")
        elif _regime == Regime.LOW and direction == "short_vol":
            confidence *= 0.80   # low vol can stay low longer but spike is nearby
            reasons.append("Regime LOW — confidence trimmed (spike proximity)")

        confidence = float(np.clip(confidence, 0.0, 1.0))

        # --- Twist 5: Kelly sizing ---
        # Short vol: high win rate (premium decays), but losses are large tail events.
        #   Need win_rate > 1/(1 + avg_win/avg_loss) for positive Kelly.
        #   With 2:1 loss:win ratio, need win_rate > 0.67.
        # Long vol: low win rate but winners are large (convex payoff).
        if direction == "short_vol":
            win_rate = 0.72
            avg_win  = abs(_spread) * 0.8 if abs(_spread) > 0 else 0.02
            avg_loss = avg_win * 2.0
        else:
            win_rate = 0.40
            avg_win  = abs(_spread) * 4.0 if abs(_spread) > 0 else 0.08
            avg_loss = avg_win * 0.35
        kf = kelly_fraction(win_rate, avg_win, avg_loss,
                            skew=float(_skew) if not pd.isna(_skew) else 0.0,
                            kurt=float(_kurt) if not pd.isna(_kurt) else 3.0)
        kf *= confidence            # scale by signal confidence

        signals.append({
            "date": idx,
            "direction": direction,
            "confidence": round(confidence, 4),
            "regime": _regime,
            "kelly_fraction": round(kf, 4),
            "rv": round(_rv, 4),
            "iv": round(implied_vol.iloc[i], 4),
            "rv_iv_spread": round(_spread, 4),
            "mtf_score": _mtf,
            "vvol_z": round(_vvol_z, 4),
            "atr": round(atr_s.iloc[i], 4),
            "upper_band": round(upper.iloc[i], 4),
            "lower_band": round(lower.iloc[i], 4),
            "reasons": "; ".join(reasons),
        })

    return pd.DataFrame(signals).set_index("date")


# ---------------------------------------------------------------------------
# Dashboard helpers
# ---------------------------------------------------------------------------

def build_signal_network(result: pd.DataFrame) -> dict:
    """
    Use NetworkX to compute pairwise correlations between model indicators,
    then export a G6-ready node/edge structure.

    Nodes  = indicators + final signal
    Edges  = correlation strength (only edges above threshold are shown)
    """
    import networkx as nx

    indicators = {
        "rv":           "Realized Vol",
        "iv":           "Implied Vol",
        "rv_iv_spread": "RV-IV Spread",
        "mtf_score":    "MTF Confluence",
        "vvol_z":       "Vol-of-Vol Z",
        "kelly_fraction": "Kelly Size",
        "confidence":   "Confidence",
    }

    numeric = result[list(indicators.keys())].dropna()
    corr    = numeric.corr()

    G = nx.Graph()

    # Add indicator nodes
    node_colors = {
        "rv":             "#4ECDC4",
        "iv":             "#FF6B6B",
        "rv_iv_spread":   "#FFE66D",
        "mtf_score":      "#A8E6CF",
        "vvol_z":         "#C3A6FF",
        "kelly_fraction": "#1ABC9C",
        "confidence":     "#F39C12",
    }
    for key, label in indicators.items():
        last_val = float(numeric[key].iloc[-1])
        G.add_node(key, label=label, color=node_colors[key],
                   value=round(last_val, 4), group="indicator")

    # Central signal node
    last_dir = result["direction"].iloc[-1]
    sig_color = {"long_vol": "#2ECC71", "short_vol": "#E74C3C", "flat": "#95A5A6"}[last_dir]
    G.add_node("signal", label=f"Signal\n{last_dir}", color=sig_color,
               value=last_dir, group="signal")

    # Correlation edges between indicators (threshold 0.3)
    keys = list(indicators.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            c = corr.loc[keys[i], keys[j]]
            if abs(c) >= 0.3:
                G.add_edge(keys[i], keys[j],
                           weight=round(abs(c), 3),
                           color="#E74C3C" if c < 0 else "#2ECC71")

    # Connect every indicator to the central signal node
    for key in indicators:
        G.add_edge(key, "signal", weight=0.5, color="#BDC3C7")

    # Layout using spring algorithm
    pos = nx.spring_layout(G, seed=42, k=2.5)

    nodes_out = []
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        nodes_out.append({
            "id":    node,
            "label": data["label"],
            "color": data["color"],
            "x":     round(x * 300 + 400, 1),
            "y":     round(y * 300 + 300, 1),
            "size":  40 if data["group"] == "signal" else 28,
            "value": str(data["value"]),
            "group": data["group"],
        })

    edges_out = []
    for u, v, data in G.edges(data=True):
        edges_out.append({
            "source": u,
            "target": v,
            "weight": data["weight"],
            "color":  data["color"],
        })

    return {"nodes": nodes_out, "edges": edges_out}


def build_dashboard_html(result, close, sharpe, max_dd, network_data) -> str:
    """Generate a self-contained HTML file using G6 for the network graph
    and SVG/Canvas charts for the time series panels."""

    import json

    dates      = [str(d.date()) for d in result.index]
    prices     = [round(float(v), 2) for v in close.reindex(result.index).values]
    rv_vals    = [round(float(v) * 100, 3) for v in result["rv"]]
    iv_vals    = [round(float(v) * 100, 3) for v in result["iv"]]
    kelly_vals = [round(float(v) * 100, 3) for v in result["kelly_fraction"]]
    vvol_vals  = [round(float(v), 3) for v in result["vvol_z"]]
    conf_vals  = [round(float(v), 3) for v in result["confidence"]]

    directions = list(result["direction"])
    regimes    = [str(r) for r in result["regime"]]

    counts     = result["direction"].value_counts().to_dict()
    long_c     = counts.get("long_vol", 0)
    short_c    = counts.get("short_vol", 0)
    flat_c     = counts.get("flat", 0)

    last       = result.iloc[-1]
    last_dir   = last["direction"]
    last_conf  = round(float(last["confidence"]) * 100, 1)
    last_kelly = round(float(last["kelly_fraction"]) * 100, 1)
    last_rv    = round(float(last["rv"]) * 100, 1)
    last_iv    = round(float(last["iv"]) * 100, 1)
    last_vvol  = round(float(last["vvol_z"]), 2)
    last_regime = str(last["regime"])
    period_start = str(result.index[0].date())
    period_end   = str(result.index[-1].date())

    dir_badge_color = {"long_vol": "#2ECC71", "short_vol": "#E74C3C", "flat": "#95A5A6"}[last_dir]

    network_json = json.dumps(network_data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Volatility Trading Model — Dashboard</title>
<script src="https://unpkg.com/@antv/g6@5/dist/g6.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0d1117; color:#e6edf3; font-family:'Segoe UI',system-ui,sans-serif; }}

  header {{
    background:linear-gradient(135deg,#161b22,#21262d);
    padding:20px 32px;
    border-bottom:1px solid #30363d;
    display:flex; align-items:center; justify-content:space-between;
  }}
  header h1 {{ font-size:20px; font-weight:700; letter-spacing:.5px; }}
  header .sub {{ font-size:12px; color:#8b949e; margin-top:4px; }}
  .badge {{
    padding:6px 14px; border-radius:20px; font-size:13px; font-weight:600;
    background:{dir_badge_color}22; color:{dir_badge_color};
    border:1px solid {dir_badge_color}55;
  }}

  .stats-row {{
    display:grid; grid-template-columns:repeat(6,1fr); gap:12px;
    padding:16px 32px;
  }}
  .stat-card {{
    background:#161b22; border:1px solid #30363d; border-radius:10px;
    padding:14px 16px;
  }}
  .stat-card .label {{ font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:.8px; }}
  .stat-card .value {{ font-size:22px; font-weight:700; margin-top:4px; }}
  .stat-card .sub   {{ font-size:11px; color:#8b949e; margin-top:2px; }}

  .grid {{
    display:grid;
    grid-template-columns:1fr 1fr;
    grid-template-rows:auto auto auto;
    gap:16px;
    padding:0 32px 32px;
  }}
  .panel {{
    background:#161b22; border:1px solid #30363d; border-radius:12px;
    padding:20px;
  }}
  .panel.wide {{ grid-column:1 / -1; }}
  .panel h2 {{ font-size:13px; font-weight:600; color:#8b949e; text-transform:uppercase;
               letter-spacing:.8px; margin-bottom:14px; }}

  canvas {{ width:100% !important; }}
  #network-container {{ height:420px; border-radius:8px; overflow:hidden; }}

  .legend {{ display:flex; gap:16px; flex-wrap:wrap; margin-top:10px; }}
  .legend-item {{ display:flex; align-items:center; gap:6px; font-size:11px; color:#8b949e; }}
  .legend-dot {{ width:10px; height:10px; border-radius:50%; flex-shrink:0; }}

  footer {{
    text-align:center; padding:16px; color:#484f58; font-size:11px;
    border-top:1px solid #21262d;
  }}
</style>
</head>
<body>

<header>
  <div>
    <h1>Volatility Trading Model</h1>
    <div class="sub">SPY &nbsp;|&nbsp; {period_start} &rarr; {period_end} &nbsp;|&nbsp; {len(result):,} bars</div>
  </div>
  <div class="badge">{last_dir.replace('_',' ').upper()}</div>
</header>

<div class="stats-row">
  <div class="stat-card">
    <div class="label">Current Signal</div>
    <div class="value" style="color:{dir_badge_color}">{last_dir.replace('_',' ').title()}</div>
    <div class="sub">Conf: {last_conf}%</div>
  </div>
  <div class="stat-card">
    <div class="label">Kelly Size</div>
    <div class="value" style="color:#1ABC9C">{last_kelly}%</div>
    <div class="sub">of capital</div>
  </div>
  <div class="stat-card">
    <div class="label">Realized Vol</div>
    <div class="value" style="color:#4ECDC4">{last_rv}%</div>
    <div class="sub">21-day annualized</div>
  </div>
  <div class="stat-card">
    <div class="label">Implied Vol (VIX)</div>
    <div class="value" style="color:#FF6B6B">{last_iv}%</div>
    <div class="sub">IV premium: {round(last_iv-last_rv,1)}%</div>
  </div>
  <div class="stat-card">
    <div class="label">Vol-of-Vol Z</div>
    <div class="value" style="color:#C3A6FF">{last_vvol}</div>
    <div class="sub">Flat if &gt; 1.5</div>
  </div>
  <div class="stat-card">
    <div class="label">Regime</div>
    <div class="value" style="color:#FFE66D">{last_regime.split('.')[-1]}</div>
    <div class="sub">Sharpe: {round(sharpe,2)}</div>
  </div>
</div>

<div class="grid">

  <!-- Price + signals -->
  <div class="panel wide">
    <h2>SPY Price &amp; Trade Signals</h2>
    <canvas id="priceChart" height="90"></canvas>
    <div class="legend">
      <div class="legend-item"><div class="legend-dot" style="background:#4ECDC4"></div>SPY Price</div>
      <div class="legend-item"><div class="legend-dot" style="background:#2ECC71"></div>Long Vol</div>
      <div class="legend-item"><div class="legend-dot" style="background:#E74C3C"></div>Short Vol</div>
    </div>
  </div>

  <!-- RV vs IV -->
  <div class="panel wide">
    <h2>Realized Vol vs Implied Vol (%)</h2>
    <canvas id="volChart" height="80"></canvas>
    <div class="legend">
      <div class="legend-item"><div class="legend-dot" style="background:#4ECDC4"></div>Realized Vol</div>
      <div class="legend-item"><div class="legend-dot" style="background:#FF6B6B"></div>Implied Vol (VIX)</div>
    </div>
  </div>

  <!-- G6 Network -->
  <div class="panel">
    <h2>Signal Flow Network (G6)</h2>
    <div id="network-container"></div>
    <div class="legend" style="margin-top:12px">
      <div class="legend-item"><div class="legend-dot" style="background:#2ECC71"></div>Positive correlation</div>
      <div class="legend-item"><div class="legend-dot" style="background:#E74C3C"></div>Negative correlation</div>
      <div class="legend-item"><div class="legend-dot" style="background:#BDC3C7"></div>Feeds into signal</div>
    </div>
  </div>

  <!-- Vol-of-Vol + Kelly -->
  <div class="panel">
    <h2>Vol-of-Vol Z-Score &amp; Kelly Size</h2>
    <canvas id="vvolChart" height="160"></canvas>
    <canvas id="kellyChart" height="100" style="margin-top:12px"></canvas>
  </div>

  <!-- Signal distribution donut -->
  <div class="panel">
    <h2>Signal Distribution</h2>
    <canvas id="donutChart" height="220"></canvas>
  </div>

  <!-- Confidence over time -->
  <div class="panel">
    <h2>Signal Confidence Over Time</h2>
    <canvas id="confChart" height="220"></canvas>
  </div>

</div>

<footer>
  Built with G6 (AntV) &bull; NetworkX &bull; Python &bull; yfinance &nbsp;&mdash;&nbsp;
  Data: Yahoo Finance &nbsp;|&nbsp; Sharpe {round(sharpe,2)} &nbsp;|&nbsp; Max DD {round(max_dd*100,1)}%
</footer>

<script>
// ── Raw data from Python ───────────────────────────────────────────────────
const DATES      = {json.dumps(dates)};
const PRICES     = {json.dumps(prices)};
const RV         = {json.dumps(rv_vals)};
const IV         = {json.dumps(iv_vals)};
const KELLY      = {json.dumps(kelly_vals)};
const VVOL       = {json.dumps(vvol_vals)};
const CONF       = {json.dumps(conf_vals)};
const DIRECTIONS = {json.dumps(directions)};
const NETWORK    = {network_json};

// Thin the data to every 5th bar so charts aren't overcrowded
const thin = (arr, n=5) => arr.filter((_,i) => i % n === 0);
const tDates  = thin(DATES);
const tPrices = thin(PRICES);
const tRV     = thin(RV);
const tIV     = thin(IV);
const tKelly  = thin(KELLY);
const tVVOL   = thin(VVOL);
const tConf   = thin(CONF);
const tDirs   = thin(DIRECTIONS);

// ── Tiny chart renderer (no external lib needed) ───────────────────────────
function sparkline(canvasId, datasets, labels, opts={{}}) {{
  const canvas = document.getElementById(canvasId);
  const ctx    = canvas.getContext('2d');
  const W = canvas.parentElement.clientWidth - 40;
  const H = canvas.height || 160;
  canvas.width  = W;
  canvas.height = H;

  const pad = {{ t:10, r:10, b:30, l:50 }};
  const cw = W - pad.l - pad.r;
  const ch = H - pad.t - pad.b;

  // Background
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, W, H);

  // Compute global min/max across all datasets
  const allVals = datasets.flatMap(d => d.data);
  const yMin = opts.yMin ?? (Math.min(...allVals) * 0.97);
  const yMax = opts.yMax ?? (Math.max(...allVals) * 1.03);
  const n    = labels.length;

  const xOf = i => pad.l + (i / (n - 1)) * cw;
  const yOf = v => pad.t + ch - ((v - yMin) / (yMax - yMin)) * ch;

  // Grid lines
  ctx.strokeStyle = '#21262d';
  ctx.lineWidth   = 1;
  for (let g = 0; g <= 4; g++) {{
    const y = pad.t + (g / 4) * ch;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + cw, y); ctx.stroke();
    const val = yMax - (g / 4) * (yMax - yMin);
    ctx.fillStyle = '#484f58'; ctx.font = '10px Segoe UI';
    ctx.fillText(val.toFixed(1), 4, y + 4);
  }}

  // Threshold line (vvol)
  if (opts.threshold !== undefined) {{
    const ty = yOf(opts.threshold);
    ctx.save();
    ctx.strokeStyle = '#E74C3C'; ctx.lineWidth = 1;
    ctx.setLineDash([4,4]);
    ctx.beginPath(); ctx.moveTo(pad.l, ty); ctx.lineTo(pad.l + cw, ty); ctx.stroke();
    ctx.restore();
  }}

  // Draw each dataset
  datasets.forEach(ds => {{
    ctx.save();

    if (ds.fill) {{
      // Gradient fill
      const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + ch);
      grad.addColorStop(0, ds.color + '55');
      grad.addColorStop(1, ds.color + '00');
      ctx.beginPath();
      ctx.moveTo(xOf(0), yOf(ds.data[0]));
      ds.data.forEach((v,i) => ctx.lineTo(xOf(i), yOf(v)));
      ctx.lineTo(xOf(ds.data.length-1), pad.t + ch);
      ctx.lineTo(xOf(0), pad.t + ch);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();
    }}

    ctx.strokeStyle = ds.color;
    ctx.lineWidth   = ds.width ?? 1.5;
    ctx.lineJoin    = 'round';
    ctx.beginPath();
    ds.data.forEach((v,i) => i === 0 ? ctx.moveTo(xOf(i), yOf(v)) : ctx.lineTo(xOf(i), yOf(v)));
    ctx.stroke();
    ctx.restore();
  }});

  // Signal dots on price chart
  if (opts.signals) {{
    tDirs.forEach((dir, i) => {{
      if (dir === 'flat') return;
      const color = dir === 'long_vol' ? '#2ECC71' : '#E74C3C';
      ctx.beginPath();
      ctx.arc(xOf(i), yOf(datasets[0].data[i]), 3, 0, Math.PI*2);
      ctx.fillStyle = color;
      ctx.fill();
    }});
  }}

  // X-axis labels (every ~10th)
  ctx.fillStyle = '#484f58'; ctx.font = '9px Segoe UI';
  const step = Math.max(1, Math.floor(n / 8));
  for (let i = 0; i < n; i += step) {{
    ctx.fillText(labels[i].slice(2,7), xOf(i) - 10, H - 8);
  }}
}}

// ── Donut chart ───────────────────────────────────────────────────────────
function donut(canvasId, data) {{
  const canvas = document.getElementById(canvasId);
  const ctx    = canvas.getContext('2d');
  const W = canvas.parentElement.clientWidth - 40;
  canvas.width = W; canvas.height = 220;
  ctx.fillStyle = '#0d1117'; ctx.fillRect(0,0,W,220);

  const cx = W/2, cy = 105, r = 80, ir = 50;
  const total = data.reduce((s,d) => s+d.value, 0);
  let angle = -Math.PI/2;

  data.forEach(d => {{
    const sweep = (d.value/total) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(cx,cy);
    ctx.arc(cx,cy,r,angle,angle+sweep);
    ctx.closePath();
    ctx.fillStyle = d.color;
    ctx.fill();

    // Label
    const mid = angle + sweep/2;
    const lx  = cx + (r+18)*Math.cos(mid);
    const ly  = cy + (r+18)*Math.sin(mid);
    ctx.fillStyle = '#e6edf3'; ctx.font = '11px Segoe UI'; ctx.textAlign='center';
    ctx.fillText(d.label, lx, ly);
    ctx.fillText(`${{d.value}}`, lx, ly+13);
    angle += sweep;
  }});

  // Hole
  ctx.beginPath(); ctx.arc(cx,cy,ir,0,Math.PI*2);
  ctx.fillStyle = '#161b22'; ctx.fill();

  // Center text
  ctx.fillStyle = '#e6edf3'; ctx.font = 'bold 14px Segoe UI'; ctx.textAlign='center';
  ctx.fillText(total, cx, cy+5);
  ctx.fillStyle = '#8b949e'; ctx.font = '10px Segoe UI';
  ctx.fillText('total bars', cx, cy+18);
}}

// ── Build charts ──────────────────────────────────────────────────────────
window.addEventListener('load', () => {{

  sparkline('priceChart',
    [{{ data: tPrices, color:'#4ECDC4', width:1.5, fill:true }}],
    tDates, {{ signals: true }});

  sparkline('volChart',
    [
      {{ data: tRV, color:'#4ECDC4', width:1.5 }},
      {{ data: tIV, color:'#FF6B6B', width:1.5 }},
    ],
    tDates);

  sparkline('vvolChart',
    [{{ data: tVVOL, color:'#C3A6FF', width:1.5, fill:true }}],
    tDates, {{ threshold: 1.5 }});

  sparkline('kellyChart',
    [{{ data: tKelly, color:'#1ABC9C', width:1.5, fill:true }}],
    tDates, {{ yMin:0 }});

  sparkline('confChart',
    [{{ data: tConf, color:'#F39C12', width:1.5, fill:true }}],
    tDates, {{ yMin:0, yMax:1 }});

  donut('donutChart', [
    {{ label:'Short Vol', value:{short_c}, color:'#E74C3C' }},
    {{ label:'Long Vol',  value:{long_c},  color:'#2ECC71' }},
    {{ label:'Flat',      value:{flat_c},  color:'#484f58' }},
  ]);

  // ── G6 Signal Network ──────────────────────────────────────────────────
  const container = document.getElementById('network-container');

  const graph = new G6.Graph({{
    container,
    width:  container.offsetWidth,
    height: 420,
    background: '#0d1117',
    node: {{
      style: {{
        size: d => d.size ?? 28,
        fill: d => d.color ?? '#4ECDC4',
        stroke: d => d.color ?? '#4ECDC4',
        fillOpacity: 0.85,
        lineWidth: 2,
        labelText: d => d.label,
        labelFill: '#e6edf3',
        labelFontSize: 11,
        labelFontFamily: 'Segoe UI',
        labelPlacement: 'bottom',
        labelOffsetY: 4,
      }},
    }},
    edge: {{
      style: {{
        stroke: e => e.color ?? '#30363d',
        lineWidth: e => Math.max(1, (e.weight ?? 0.3) * 3),
        strokeOpacity: 0.7,
        endArrow: false,
      }},
    }},
    behaviors: ['drag-canvas','zoom-canvas','drag-element'],
    plugins: [{{
      type: 'tooltip',
      getContent: (e, items) => {{
        const d = items[0]?.data;
        if (!d) return '';
        return `<div style="background:#21262d;padding:8px 12px;border-radius:6px;
                            border:1px solid #30363d;font-size:12px;color:#e6edf3">
                  <b>${{d.label}}</b><br/>Current: ${{d.value}}
                </div>`;
      }},
    }}],
    data: {{
      nodes: NETWORK.nodes.map(n => ({{
        id: n.id, label: n.label, color: n.color,
        x: n.x, y: n.y, size: n.size, value: n.value, group: n.group,
      }})),
      edges: NETWORK.edges.map((e,i) => ({{
        id: `e${{i}}`, source: e.source, target: e.target,
        weight: e.weight, color: e.color,
      }})),
    }},
  }});

  graph.render();
}});
</script>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json, webbrowser, pathlib, yfinance as yf

    print("Fetching real market data from Yahoo Finance...")

    spy = yf.download("SPY", start="2020-01-01", auto_adjust=True, progress=False)
    vix = yf.download("^VIX", start="2020-01-01", auto_adjust=True, progress=False)

    shared = spy.index.intersection(vix.index)
    spy    = spy.loc[shared]
    vix    = vix.loc[shared]

    close = spy["Close"].squeeze()
    high  = spy["High"].squeeze()
    low   = spy["Low"].squeeze()
    iv    = (vix["Close"] / 100).squeeze()

    print(f"Loaded {len(close)} bars of SPY + VIX data "
          f"({close.index[0].date()} to {close.index[-1].date()})")

    result = generate_signal(close, high, low, iv)

    # Backtest metrics
    result["ret"] = close.pct_change()

    def trade_return(row):
        if row.direction == "long_vol":
            return abs(row.ret) - row.rv / 252
        elif row.direction == "short_vol":
            return row.rv / 252 - abs(row.ret)
        return 0.0

    result["pnl"]    = result.apply(trade_return, axis=1) * result["kelly_fraction"]
    result["equity"] = (1 + result["pnl"]).cumprod()

    sharpe_num = result["pnl"].mean()
    sharpe_den = result["pnl"].std()
    sharpe     = (sharpe_num / sharpe_den * np.sqrt(252)) if sharpe_den > 0 else 0
    max_dd     = ((result["equity"] / result["equity"].cummax()) - 1).min()

    print(f"Signal counts: {result.direction.value_counts().to_dict()}")
    print(f"Sharpe: {sharpe:.2f}  |  Max DD: {max_dd*100:.1f}%")
    print("Building dashboard...")

    network_data = build_signal_network(result)
    html         = build_dashboard_html(result, close, sharpe, max_dd, network_data)

    out_path = pathlib.Path(__file__).parent / "dashboard.html"
    out_path.write_text(html, encoding="utf-8")

    print(f"Dashboard saved to {out_path}")
    print("Opening in browser...")
    webbrowser.open(out_path.as_uri())
