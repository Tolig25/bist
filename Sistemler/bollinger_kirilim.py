"""
Strateji: Bollinger Bands Kırılım
Fiyatın Bollinger Bantları dışına çıkmasını ve geri dönüşünü takip eder.
"""

NAME        = "Bollinger Bantları Kırılım"
DESCRIPTION = "BB sıkışması ve kırılım noktalarını tespit eder."

import pandas as pd


def analyze(df: pd.DataFrame) -> dict:
    try:
        import pandas_ta as ta
        df = df.copy()
        df.ta.bbands(length=20, std=2, append=True)
    except ImportError:
        return {"hint": "pandas_ta yüklü değil", "signal": "BEKLE", "score": 50}

    lower_col = "BBL_20_2.0"
    mid_col   = "BBM_20_2.0"
    upper_col = "BBU_20_2.0"
    bw_col    = "BBB_20_2.0"

    if upper_col not in df.columns:
        return {"hint": "Bollinger bantları hesaplanamadı", "signal": "BEKLE", "score": 50}

    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    close = float(last["Close"])
    upper = float(last[upper_col])
    lower = float(last[lower_col])
    mid   = float(last[mid_col])

    score  = 50
    signal = "BEKLE"
    notes  = []

    # Bant genişliği (sıkışma kontrolü)
    bw_pct = ((upper - lower) / mid) * 100 if mid > 0 else 0
    if bw_pct < 5:
        notes.append(f"BB sıkışması tespit edildi (%{bw_pct:.1f} genişlik) – kırılım beklenebilir")

    # Fiyat konumu
    if close < lower:
        score += 30
        signal = "AL"
        notes.append(f"Fiyat alt bandın altında ({close:.2f} < {lower:.2f}) – aşırı satım")
    elif close > upper:
        score -= 30
        signal = "SAT"
        notes.append(f"Fiyat üst bandın üstünde ({close:.2f} > {upper:.2f}) – aşırı alım")
    elif close > mid:
        score += 10
        notes.append(f"Fiyat orta bandın üzerinde, pozitif eğilim")
    else:
        score -= 10
        notes.append(f"Fiyat orta bandın altında, negatif eğilim")

    # Önceki mumda aşağı kırılım geri dönüşü (Reversal)
    prev_close = float(prev["Close"])
    prev_lower = float(prev[lower_col])
    if prev_close < prev_lower and close > lower:
        score += 15
        notes.append("Alt banttan geri dönüş başladı (güçlü AL sinyali)")

    score = max(0, min(100, score))
    if score >= 65:
        signal = "AL"
    elif score <= 35:
        signal = "SAT"

    return {
        "hint":        " | ".join(notes),
        "signal":      signal,
        "score":       score,
        "bb_upper":    round(upper, 2),
        "bb_lower":    round(lower, 2),
        "bb_mid":      round(mid, 2),
        "bandwidth_pct": round(bw_pct, 2),
    }
