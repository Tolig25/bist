import numpy as np

class RiskManager:
    @staticmethod
    def calculate_autonomous_risk(df):
        """Hissenin oynaklığına göre manipülasyon riski üretir."""
        returns = df['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252) * 100 # Yıllıklandırılmış volatilite
        
        # Risk Skoru (0-100 arası)
        # Genelde %40 üstü volatilite BIST için 'spekülatif' kabul edilebilir.
        risk_score = min(volatility * 1.5, 100)
        
        return {
            "volatility": round(volatility, 2),
            "risk_score": round(risk_score, 1),
            "is_manipulative": risk_score > 60
        }
