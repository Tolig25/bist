import os
import gc
import json
import time
import threading
import importlib
import traceback
from pathlib import Path
from flask import Flask, jsonify, request, render_template_string, send_from_directory

import yfinance as yf
import pandas as pd

app = Flask(__name__)

# ─── LLM Manager ────────────────────────────────────────────────────────────

class LLM_Manager:
    """
    Manages a GGUF model via llama-cpp-python.
    Model is downloaded once from HuggingFace Hub on first use.
    """
    _instance = None
    _lock = threading.Lock()

    MODEL_REPO   = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    MODEL_FILE   = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    LOCAL_DIR    = Path("./models")
    CONTEXT_SIZE = 2048
    MAX_TOKENS   = 512

    def __init__(self):
        self.llm = None
        self.loaded = False
        self.loading = False
        self.error = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _download_model(self) -> Path:
        """Download model from HuggingFace Hub if not present."""
        self.LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        local_path = self.LOCAL_DIR / self.MODEL_FILE
        if local_path.exists():
            print(f"[LLM] Model already cached at {local_path}")
            return local_path

        print(f"[LLM] Downloading {self.MODEL_FILE} from {self.MODEL_REPO} ...")
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=self.MODEL_REPO,
                filename=self.MODEL_FILE,
                local_dir=str(self.LOCAL_DIR),
                local_dir_use_symlinks=False,
            )
            print(f"[LLM] Download complete: {path}")
            return Path(path)
        except Exception as e:
            raise RuntimeError(f"Model download failed: {e}")

    def load(self):
        """Load model into memory (call once, thread-safe)."""
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            self.loading = True
            try:
                model_path = self._download_model()
                from llama_cpp import Llama
                print("[LLM] Loading model into memory …")
                self.llm = Llama(
                    model_path=str(model_path),
                    n_ctx=self.CONTEXT_SIZE,
                    n_threads=int(os.environ.get("LLM_THREADS", "2")),
                    n_gpu_layers=0,           # CPU only on Render
                    verbose=False,
                )
                self.loaded = True
                print("[LLM] Model loaded successfully.")
            except Exception as e:
                self.error = str(e)
                print(f"[LLM] Load error: {e}")
                traceback.print_exc()
            finally:
                self.loading = False

    def generate(self, prompt: str) -> str:
        """Run inference. Returns raw text."""
        if not self.loaded:
            self.load()
        if not self.loaded:
            return f"[Model not available: {self.error}]"

        gc.collect()
        try:
            output = self.llm(
                prompt,
                max_tokens=self.MAX_TOKENS,
                temperature=0.3,
                top_p=0.9,
                stop=["```", "---", "\n\n\n"],
                echo=False,
            )
            gc.collect()
            return output["choices"][0]["text"].strip()
        except Exception as e:
            gc.collect()
            return f"[Inference error: {e}]"


llm_manager = LLM_Manager.get_instance()

# ─── Sistem (Strategy) Loader ────────────────────────────────────────────────

SYSTEMS_DIR = Path("./sistemler")

def load_systems():
    """
    Scan sistemler/ and import every .py file.
    Each module must expose:
        NAME        : str  – display name
        DESCRIPTION : str  – short description
        def analyze(df: pd.DataFrame) -> dict
    """
    systems = {}
    SYSTEMS_DIR.mkdir(exist_ok=True)
    for fpath in sorted(SYSTEMS_DIR.glob("*.py")):
        mod_name = fpath.stem
        try:
            spec = importlib.util.spec_from_file_location(mod_name, fpath)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            systems[mod_name] = {
                "name":        getattr(mod, "NAME",        mod_name),
                "description": getattr(mod, "DESCRIPTION", ""),
                "module":      mod,
            }
        except Exception as e:
            print(f"[SYSTEM] Could not load {fpath}: {e}")
    return systems

# ─── Technical Analysis Helper ───────────────────────────────────────────────

def fetch_ohlcv(ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    ticker = ticker.upper()
    if not ticker.endswith(".IS"):
        ticker += ".IS"
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    df.dropna(inplace=True)
    return df

def compute_indicators(df: pd.DataFrame) -> dict:
    """Compute RSI, MACD, ATR using pandas_ta."""
    try:
        import pandas_ta as ta
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.atr(length=14, append=True)
    except ImportError:
        pass

    close = df["Close"].iloc[-1]
    rsi   = df.get("RSI_14",    pd.Series([50])).iloc[-1]
    macd  = df.get("MACD_12_26_9", pd.Series([0])).iloc[-1]
    macd_s= df.get("MACDs_12_26_9", pd.Series([0])).iloc[-1]
    atr   = df.get("ATRr_14",   pd.Series([close * 0.02])).iloc[-1]

    highs = df["High"].tail(50)
    lows  = df["Low"].tail(50)
    resistance = round(float(highs.max()), 2)
    support    = round(float(lows.min()), 2)

    return {
        "close":      round(float(close), 2),
        "rsi":        round(float(rsi), 2),
        "macd":       round(float(macd), 4),
        "macd_signal":round(float(macd_s), 4),
        "atr":        round(float(atr), 4),
        "support":    support,
        "resistance": resistance,
    }

def build_prompt(ticker: str, indicators: dict, strategy_hint: str = "") -> str:
    hint_text = f"\nStrateji bağlamı: {strategy_hint}" if strategy_hint else ""
    return f"""Sen bir profesyonel borsa analistisisin. Türkiye BIST hisse senedi analizi yapıyorsun.

Hisse: {ticker}
Teknik Göstergeler:
- Son Kapanış: {indicators['close']} TL
- RSI (14): {indicators['rsi']}
- MACD: {indicators['macd']} | Sinyal: {indicators['macd_signal']}
- ATR (14): {indicators['atr']}
- Destek: {indicators['support']} TL
- Direnç: {indicators['resistance']} TL{hint_text}

Aşağıdaki JSON formatında analiz yap (başka hiçbir şey yazma):
{{
  "skor": <0-100 arası sayı>,
  "sinyal": "<AL veya SAT veya BEKLE>",
  "alis_limiti": <sayı>,
  "satis_limiti": <sayı>,
  "stop_loss": <sayı>,
  "destek": <sayı>,
  "direnc": <sayı>,
  "beklenen_vade_gun": <sayı>,
  "yorum": "<kısa Türkçe yorum>"
}}"""

def parse_llm_json(raw: str) -> dict:
    """Extract JSON from LLM output robustly."""
    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except Exception:
        pass
    return {
        "skor": 50, "sinyal": "BEKLE",
        "alis_limiti": 0, "satis_limiti": 0, "stop_loss": 0,
        "destek": 0, "direnc": 0, "beklenen_vade_gun": 0,
        "yorum": raw[:200],
    }

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    html_path = Path("index.html")
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>index.html bulunamadı</h1>", 404

@app.route("/api/systems")
def api_systems():
    systems = load_systems()
    return jsonify([
        {"id": k, "name": v["name"], "description": v["description"]}
        for k, v in systems.items()
    ])

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data      = request.get_json(force=True)
    ticker    = data.get("ticker", "THYAO").upper()
    system_id = data.get("system", None)
    period    = data.get("period", "3mo")

    try:
        df = fetch_ohlcv(ticker, period=period)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    indicators = compute_indicators(df)

    # Run optional strategy module
    strategy_hint = ""
    strategy_extra = {}
    systems = load_systems()
    if system_id and system_id in systems:
        try:
            strategy_extra = systems[system_id]["module"].analyze(df)
            strategy_hint  = strategy_extra.get("hint", "")
        except Exception as e:
            strategy_hint = f"Strateji hatası: {e}"

    prompt = build_prompt(ticker, indicators, strategy_hint)
    raw    = llm_manager.generate(prompt)
    result = parse_llm_json(raw)

    # Candle data for chart (last 90 days)
    candles = []
    for ts, row in df.tail(90).iterrows():
        candles.append({
            "time":  int(pd.Timestamp(ts).timestamp()),
            "open":  round(float(row["Open"]), 2),
            "high":  round(float(row["High"]), 2),
            "low":   round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
        })

    return jsonify({
        "ticker":     ticker,
        "indicators": indicators,
        "analysis":   result,
        "candles":    candles,
        "strategy":   strategy_extra,
        "llm_raw":    raw,
    })

@app.route("/api/llm_status")
def api_llm_status():
    return jsonify({
        "loaded":  llm_manager.loaded,
        "loading": llm_manager.loading,
        "error":   llm_manager.error,
    })

@app.route("/api/warmup", methods=["POST"])
def api_warmup():
    """Trigger model load in background."""
    if not llm_manager.loaded and not llm_manager.loading:
        t = threading.Thread(target=llm_manager.load, daemon=True)
        t.start()
    return jsonify({"status": "warming_up"})

# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
