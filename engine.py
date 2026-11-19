import pandas as pd
import numpy as np

ALIASES = {
    "time": ["time", "date", "datetime", "<date>", "timestamp"],
    "open": ["open", "<open>"], "high": ["high", "<high>"],
    "low": ["low", "<low>"], "close": ["close", "<close>"],
    "volume": ["tick_volume", "volume", "real_volume", "<tickvol>", "<vol>"],
}

def normalize_columns(raw):
    if raw.empty:
        raise ValueError("The CSV contains no rows.")
    lookup = {str(c).strip().lower(): c for c in raw.columns}
    rename = {}
    for target, aliases in ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                rename[lookup[alias]] = target
                break
    df = raw.rename(columns=rename).copy()
    missing = [c for c in ["open", "high", "low", "close"] if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if "time" not in df:
        df["time"] = range(len(df))
    if "volume" not in df:
        df["volume"] = 0
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if len(df) < 40:
        raise ValueError("At least 40 valid candles are required.")
    if (df.high < df.low).any():
        raise ValueError("Invalid candles found: a high is below its low.")
    return df[["time", "open", "high", "low", "close", "volume"]]

def calculate_atr(df, period=14):
    prev = df.close.shift(1)
    tr = pd.concat([(df.high-df.low), (df.high-prev).abs(), (df.low-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def confirmed_pivots(df, length=5):
    highs, lows = [], []
    # Pivot at p is confirmed only when `length` later candles are present.
    for p in range(length, len(df)-length):
        hw = df.high.iloc[p-length:p+length+1]
        lw = df.low.iloc[p-length:p+length+1]
        if df.high.iloc[p] == hw.max() and (hw == df.high.iloc[p]).sum() == 1:
            highs.append((p, float(df.high.iloc[p])))
        if df.low.iloc[p] == lw.min() and (lw == df.low.iloc[p]).sum() == 1:
            lows.append((p, float(df.low.iloc[p])))
    return highs, lows

def analyze_market(df):
    df = df.copy()
    df["atr14"] = calculate_atr(df)
    ph, pl = confirmed_pivots(df, 5)
    close = float(df.close.iloc[-1])
    atr_now = float(df.atr14.iloc[-1]) if pd.notna(df.atr14.iloc[-1]) else 0.0

    structure, detail = "Unclear", "Insufficient confirmed swing points."
    if len(ph) >= 2 and len(pl) >= 2:
        hh, hl = ph[-1][1] > ph[-2][1], pl[-1][1] > pl[-2][1]
        lh, ll = ph[-1][1] < ph[-2][1], pl[-1][1] < pl[-2][1]
        if hh and hl:
            structure, detail = "Bullish", "The two latest confirmed swing highs and lows form HH/HL."
        elif lh and ll:
            structure, detail = "Bearish", "The two latest confirmed swing highs and lows form LH/LL."
        else:
            structure, detail = "Mixed / range", "Recent confirmed swing highs and lows do not agree."

    bos = []
    if ph and ph[-1][0] < len(df)-1 and close > ph[-1][1]:
        bos.append({"event": "Close above latest confirmed swing high", "level": ph[-1][1], "pivot_bar": ph[-1][0]})
    if pl and pl[-1][0] < len(df)-1 and close < pl[-1][1]:
        bos.append({"event": "Close below latest confirmed swing low", "level": pl[-1][1], "pivot_bar": pl[-1][0]})

    zone_height = atr_now if atr_now > 0 else max(float((df.high-df.low).tail(20).mean()), 1e-8)
    zones = []
    for i, price in ph[-6:]:
        zones.append({"type": "Resistance candidate", "top": price, "bottom": price-zone_height, "pivot_bar": i})
    for i, price in pl[-6:]:
        zones.append({"type": "Support candidate", "top": price+zone_height, "bottom": price, "pivot_bar": i})

    gaps = []
    for i in range(max(2, len(df)-150), len(df)):
        if df.low.iloc[i] > df.high.iloc[i-2]:
            gaps.append({"type": "Bullish FVG candidate", "top": float(df.low.iloc[i]), "bottom": float(df.high.iloc[i-2]), "bar": i})
        if df.high.iloc[i] < df.low.iloc[i-2]:
            gaps.append({"type": "Bearish FVG candidate", "top": float(df.low.iloc[i-2]), "bottom": float(df.high.iloc[i]), "bar": i})

    status = "Review candidates on chart" if structure in ("Bullish", "Bearish") else "Wait / unclear structure"
    summary = f"Last close: {close:.6g}. Structure: {structure.lower()}. This is a basic candidate scan, not an entry signal."
    return {"structure": structure, "structure_detail": detail, "status": status,
            "summary": summary, "zones": zones[-12:], "bos": bos[-10:], "fvg": gaps[-12:]}
