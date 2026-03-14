import os

class Config:
    # Flask Ayarları
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'mekatronik-hedef-2026'
    
    # Borsa Ayarları
    DEFAULT_SYMBOL = "THYAO.IS"
    RISK_FREE_RATE = 0.02  # Analizlerde kullanılacak risksiz faiz oranı
    
    # Teknik Analiz Parametreleri
    RSI_PERIOD = 14
    EMA_SHORT = 12
    EMA_LONG = 26
