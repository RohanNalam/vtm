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
# Quick demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yfinance as yf

    print("Fetching real market data from Yahoo Finance...")

    spy = yf.download("SPY", start="2020-01-01", auto_adjust=True, progress=False)
    vix = yf.download("^VIX", start="2020-01-01", auto_adjust=True, progress=False)

    # Align on shared trading dates
    shared = spy.index.intersection(vix.index)
    spy = spy.loc[shared]
    vix = vix.loc[shared]

    close = spy["Close"].squeeze()
    high  = spy["High"].squeeze()
    low   = spy["Low"].squeeze()
    iv    = (vix["Close"] / 100).squeeze()  # VIX as annualized IV fraction

    print(f"Loaded {len(close)} bars of SPY + VIX data "
          f"({close.index[0].date()} to {close.index[-1].date()})")

    result = generate_signal(close, high, low, iv)

    print("=" * 70)
    print("VOLATILE TRADING MODEL — Enhanced Signal Output")
    print("=" * 70)
    print(f"\nTotal bars  : {len(result)}")
    print(f"Long vol    : {(result.direction == 'long_vol').sum()}")
    print(f"Short vol   : {(result.direction == 'short_vol').sum()}")
    print(f"Flat        : {(result.direction == 'flat').sum()}")

    print("\n--- Last 10 signals ---")
    cols = ["direction", "confidence", "regime", "kelly_fraction",
            "rv", "iv", "rv_iv_spread", "mtf_score", "vvol_z"]
    print(result[cols].tail(10).to_string())

    print("\n--- Sample reasons ---")
    sample = result[result.direction != "flat"].tail(5)
    for idx, row in sample.iterrows():
        print(f"  {idx.date()} | {row.direction:10s} | conf={row.confidence:.2f} "
              f"| kelly={row.kelly_fraction:.3f} | {row.reasons}")

    # --- Regime distribution ---
    print("\n--- Regime distribution ---")
    print(result.regime.value_counts())

    # --- Simple backtest: equity curve ---
    print("\n--- Simplified equity curve (last 20 bars) ---")
    returns_series = close.pct_change()
    result["ret"] = returns_series

    def trade_return(row):
        if row.direction == "long_vol":
            # long vol profits when abs(return) > realized vol expectation
            return abs(row.ret) - row.rv / 252
        elif row.direction == "short_vol":
            # short vol profits when abs(return) < realized vol expectation
            return row.rv / 252 - abs(row.ret)
        return 0.0

    result["pnl"] = result.apply(trade_return, axis=1) * result["kelly_fraction"]
    result["equity"] = (1 + result["pnl"]).cumprod()

    print(result[["direction", "pnl", "equity", "kelly_fraction"]].tail(20).to_string())

    total_ret  = result["equity"].iloc[-1] - 1
    sharpe_num = result["pnl"].mean()
    sharpe_den = result["pnl"].std()
    sharpe     = (sharpe_num / sharpe_den * np.sqrt(252)) if sharpe_den > 0 else 0

    print(f"\nTotal return : {total_ret * 100:.2f}%")
    print(f"Sharpe ratio : {sharpe:.2f}")
    print(f"Max drawdown : {((result['equity'] / result['equity'].cummax()) - 1).min() * 100:.2f}%")
