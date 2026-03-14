from scipy import stats
import numpy as np

class FuturePredictor:
    @staticmethod
    def linear_projection(df, days=1):
        """Son 20 günlük trend eğimine göre gelecek tahmini yapar."""
        prices = df['Close'].values[-20:]
        x = np.arange(len(prices))
        
        # Lineer Regresyon: y = mx + c
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        
        # Tahmin Edilen Fiyat
        target_index = len(prices) + days
        predicted_price = intercept + slope * target_index
        
        # Güven Aralığı (Standart Hata üzerinden)
        confidence_margin = std_err * 1.96 * np.sqrt(days)
        
        return {
            "predicted_price": round(predicted_price, 2),
            "margin": round(confidence_margin, 2),
            "trend": "Yukarı" if slope > 0 else "Aşağı",
            "accuracy": round(r_value**2, 2) # R-Kare (Modelin gücü)
        }
