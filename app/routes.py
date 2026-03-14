from flask import render_template, request, jsonify
from app import app
import yfinance as yf
from core.technical import TechnicalAnalysis
from core.prediction import FuturePredictor
from core.risk_manager import RiskManager

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['GET'])
def analyze():
    symbol = request.args.get('symbol', 'THYAO.IS').upper()
    days = int(request.args.get('days', 1))

    try:
        # Veri çekme
        raw_data = yf.download(symbol, period="1y", interval="1d")
        if raw_data.empty: return jsonify({"error": "Sembol bulunamadı."})

        # Teknik Analiz
        df = TechnicalAnalysis.get_indicators(raw_data)
        
        # Tahmin ve Risk
        prediction = FuturePredictor.linear_projection(df, days)
        risk = RiskManager.calculate_autonomous_risk(df)

        # JSON formatında paketleme
return jsonify({
        "symbol": symbol,
        "current_price": round(df['Close'].iloc[-1], 2),
        "rsi": round(df['RSI'].iloc[-1], 2),
        "prediction": prediction,
        "risk": risk,
        "chart_prices": df['Close'].tail(30).tolist(), # Son 30 günün fiyatı
        "chart_labels": df.index.strftime('%d %b').tolist()[-30:], # Son 30 günün tarihi
        "status": "success"
    })
    except Exception as e:
        return jsonify({"error": str(e)})
