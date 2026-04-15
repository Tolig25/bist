"""
Strateji: RSI + MACD Kombinasyon
RSI aşırı alım/satım bölgelerini MACD kesişimi ile teyit eder.
"""

NAME        = "RSI + MACD Kombinasyon"
DESCRIPTION = "RSI 30/70 seviyeleri ile MACD sinyal kesişimini birleştirir."

import pandas as pd


def analyze(df: pd.DataFrame) -> dict:
    """
    Returns:
        hint  : str  – LLM prompt'a eklenecek strateji özeti
        signal: str  – AL / SAT / BEKLE
        score : int  – 0-100
    """
    try:
        import pandas_ta as ta
        df = df.copy()
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
    except ImportError:
        return {"hint": "pandas_ta yüklü değil", "signal": "BEKLE", "score": 50}

    rsi_col   = "RSI_14"
    macd_col  = "MACD_12_26_9"
    macd_s    = "MACDs_12_26_9"

    if rsi_col not in df.columns:
        return {"hint": "RSI hesaplanamadı", "signal": "BEKLE", "score": 50}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    rsi  = float(last[rsi_col])
    macd_val  = float(last[macd_col]) if macd_col in df.columns else 0
    macd_sig  = float(last[macd_s])   if macd_s   in df.columns else 0
    pmacd_val = float(prev[macd_col]) if macd_col in df.columns else 0
    pmacd_sig = float(prev[macd_s])   if macd_s   in df.columns else 0

    # MACD yukarı kesişim?
    macd_cross_up   = (pmacd_val <= pmacd_sig) and (macd_val > macd_sig)
    # MACD aşağı kesişim?
    macd_cross_down = (pmacd_val >= pmacd_sig) and (macd_val < macd_sig)

    score  = 50
    signal = "BEKLE"
    notes  = []

    if rsi < 30:
        score += 20
        notes.append(f"RSI aşırı satım bölgesinde ({rsi:.1f})")
    elif rsi > 70:
        score -= 20
        notes.append(f"RSI aşırı alım bölgesinde ({rsi:.1f})")
    else:
        notes.append(f"RSI nötr ({rsi:.1f})")

    if macd_cross_up:
        score += 25
        notes.append("MACD yukarı kesişim oluştu (Alış sinyali)")
    elif macd_cross_down:
        score -= 25
        notes.append("MACD aşağı kesişim oluştu (Satış sinyali)")
    elif macd_val > macd_sig:
        score += 10
        notes.append("MACD sinyal üzerinde (pozitif momentum)")
    else:
        score -= 10
        notes.append("MACD sinyal altında (negatif momentum)")

    score = max(0, min(100, score))

    if score >= 65:
        signal = "AL"
    elif score <= 35:
        signal = "SAT"

    hint = " | ".join(notes)
    return {
        "hint":   hint,
        "signal": signal,
        "score":  score,
        "rsi":    round(rsi, 2),
        "macd_cross_up":   macd_cross_up,
        "macd_cross_down": macd_cross_down,
    }
