"""
Strateji: Hacim Trend Onayı
Yükselen/düşen trendi işlem hacmi ile teyit eder.
"""

NAME        = "Hacim Trend Onayı"
DESCRIPTION = "EMA trend yönünü hacim artışı ile doğrular."

import pandas as pd


def analyze(df: pd.DataFrame) -> dict:
    try:
        import pandas_ta as ta
        df = df.copy()
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
    except ImportError:
        return {"hint": "pandas_ta yüklü değil", "signal": "BEKLE", "score": 50}

    ema20_col = "EMA_20"
    ema50_col = "EMA_50"

    score  = 50
    signal = "BEKLE"
    notes  = []

    close = float(df["Close"].iloc[-1])

    # EMA trend
    if ema20_col in df.columns and ema50_col in df.columns:
        ema20 = float(df[ema20_col].iloc[-1])
        ema50 = float(df[ema50_col].iloc[-1])
        if ema20 > ema50:
            score += 15
            notes.append(f"EMA20 ({ema20:.2f}) > EMA50 ({ema50:.2f}) – yükselen trend")
        else:
            score -= 15
            notes.append(f"EMA20 ({ema20:.2f}) < EMA50 ({ema50:.2f}) – düşen trend")

        if close > ema20:
            score += 10
            notes.append("Fiyat EMA20 üzerinde")
        else:
            score -= 10
            notes.append("Fiyat EMA20 altında")

    # Hacim analizi
    if "Volume" in df.columns:
        vol_series = df["Volume"].dropna()
        if len(vol_series) >= 20:
            avg_vol = float(vol_series.tail(20).mean())
            last_vol = float(vol_series.iloc[-1])
            vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0

            if vol_ratio > 1.5:
                score += 20
                notes.append(f"Hacim ortalamanın {vol_ratio:.1f}x üzerinde – güçlü hareket")
            elif vol_ratio > 1.2:
                score += 10
                notes.append(f"Hacim ortalamanın üzerinde ({vol_ratio:.1f}x)")
            elif vol_ratio < 0.5:
                score -= 10
                notes.append(f"Düşük hacim – zayıf hareket ({vol_ratio:.1f}x)")

    score = max(0, min(100, score))
    if score >= 65:
        signal = "AL"
    elif score <= 35:
        signal = "SAT"

    return {
        "hint":   " | ".join(notes),
        "signal": signal,
        "score":  score,
    }
