import os
import json
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pandas as pd
import pandas_ta as ta

# --- Flaskアプリケーションの初期化 ---
app = Flask(__name__)
CORS(app) 

# --- 設定値 ---
CRYPTOCOMPARE_API_KEY = os.environ.get('CRYPTOCOMPARE_API_KEY', None)
CRYPTOCOMPARE_API_URL = 'https://min-api.cryptocompare.com/data/v2/'

# --- 【プロフェッショナル版】資産クラスごとに最適化されたパラメータ ---
analysis_configs = {
    'default': {
        'weights': { 'ma': 1.0, 'macd': 1.0, 'rsi': 1.0, 'stoch': 1.0, 'bb': 1.0, 'psar': 1.2, 'mtaConfirmation': 2.0, 'volumeConfirmation': 1.5, 'maSlope': 1.2, 'candlestick': 2.5, 'obv_divergence': 2.2, 'bb_squeeze': 3.0, 'pivot': 2.0, 'fibonacci': 1.8 },
        'params': { 'rsiPeriod': 14, 'rsiOverbought': 70, 'rsiOversold': 30, 'stochPeriod': 14, 'bbPeriod': 20, 'bbStdDev': 2, 'emaShort': 12, 'emaLong': 26, 'macdSignal': 9, 'psarStart': 0.02, 'psarIncrement': 0.02, 'psarMax': 0.2, 'signalThreshold': 4.0, 'adxPeriod': 14, 'adxThreshold': 25, 'atrPeriod': 14, 'slopePeriod': 10, 'kc_period': 20, 'kc_mult': 2, 'fib_lookback': 50 }
    },
    'forex': {
        'weights': { 'ma': 1.2, 'rsi': 1.2, 'bb': 1.2, 'maSlope': 1.5, 'candlestick': 3.0, 'pivot': 3.5, 'fibonacci': 2.5 },
        'params': { 'emaShort': 9, 'emaLong': 21, 'signalThreshold': 4.2 }
    },
    'gold': {
        'weights': { 'ma': 1.5, 'macd': 1.5, 'psar': 1.5, 'maSlope': 1.8, 'obv_divergence': 2.5, 'pivot': 3.0, 'fibonacci': 2.2 },
        'params': { 'emaShort': 15, 'emaLong': 30, 'adxThreshold': 22 }
    },
    'crypto': {
        'weights': { 'ma': 1.2, 'rsi': 0.8, 'bb': 1.5, 'volumeConfirmation': 2.0, 'bb_squeeze': 3.5 },
        'params': { 'rsiOverbought': 75, 'rsiOversold': 25, 'emaShort': 20, 'emaLong': 50, 'signalThreshold': 4.5, 'adxThreshold': 28, 'kc_mult': 2.5 }
    }
}

timeframeConfigs = {
    'minute': {'label': '1分足', 'endpoint': 'histominute', 'aggregate': 1, 'limit': 240, 'mta': [{'key': '5minute', 'weight': 0.6}, {'key': '15minute', 'weight': 0.4}]},
    '5minute': {'label': '5分足', 'endpoint': 'histominute', 'aggregate': 5, 'limit': 96, 'mta': [{'key': '15minute', 'weight': 0.7}, {'key': 'hour', 'weight': 0.3}]},
    '15minute': {'label': '15分足', 'endpoint': 'histominute', 'aggregate': 15, 'limit': 96, 'mta': [{'key': 'hour', 'weight': 0.8}, {'key': '4hour', 'weight': 0.2}]},
    'hour': {'label': '1時間足', 'endpoint': 'histohour', 'aggregate': 1, 'limit': 168, 'mta': [{'key': '4hour', 'weight': 0.7}, {'key': 'day', 'weight': 0.3}]},
    '4hour': {'label': '4時間足', 'endpoint': 'histohour', 'aggregate': 4, 'limit': 180, 'mta': [{'key': 'day', 'weight': 1.0}]},
    'day': {'label': '日足', 'endpoint': 'histoday', 'aggregate': 1, 'limit': 200, 'mta': []},
}

# --- ヘルパー関数 ---
def get_asset_class(pair_str):
    if pair_str in ['XAUT', 'XAU']: return 'gold'
    forex_bases = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'CAD', 'CHF', 'NZD']
    if len(pair_str) == 6 and pair_str[:3] in forex_bases and pair_str[3:] in forex_bases: return 'forex'
    return 'crypto'

def get_merged_config(asset_class):
    config = analysis_configs['default'].copy()
    asset_config = analysis_configs.get(asset_class, {})
    config['weights'].update(asset_config.get('weights', {}))
    config['params'].update(asset_config.get('params', {}))
    return config

def parse_pair(pair_str):
    forex_quotes = ['JPY', 'USD', 'EUR', 'GBP', 'AUD', 'CAD', 'CHF', 'NZD']
    if len(pair_str) <= 4: return {'base': pair_str, 'quote': 'USD'}
    potential_quote = pair_str[-3:]
    if len(pair_str) == 6 and potential_quote in forex_quotes:
        base = pair_str[:3]
        return {'base': base, 'quote': potential_quote}
    return {'base': pair_str, 'quote': 'USD'}

def fetch_data(symbol, tsym, endpoint, limit, aggregate):
    agg_param = f"&aggregate={aggregate}" if aggregate > 1 else ""
    url = f"{CRYPTOCOMPARE_API_URL}{endpoint}?fsym={symbol}&tsym={tsym}&limit={limit}{agg_param}"
    if CRYPTOCOMPARE_API_KEY: url += f"&api_key={CRYPTOCOMPARE_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('Response') == 'Error': return None
        df = pd.DataFrame(data['Data']['Data'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {symbol}/{tsym}: {e}")
        return None

def get_basic_signal(df, params):
    df.ta.ema(length=params['emaShort'], append=True, col_names=(f"EMA_S",))
    df.ta.ema(length=params['emaLong'], append=True, col_names=(f"EMA_L",))
    if df.empty or "EMA_S" not in df.columns or "EMA_L" not in df.columns or len(df) < 2: return 'hold'
    last, prev = df.iloc[-1], df.iloc[-2]
    if prev['EMA_S'] < prev['EMA_L'] and last['EMA_S'] > last['EMA_L']: return 'buy'
    if prev['EMA_S'] > prev['EMA_L'] and last['EMA_S'] < last['EMA_L']: return 'sell'
    return 'hold'

def detect_divergence(price_series, indicator_series, lookback=30):
    price_lows = price_series.rolling(window=lookback).min()
    price_highs = price_series.rolling(window=lookback).max()
    indicator_lows = indicator_series.rolling(window=lookback).min()
    indicator_highs = indicator_series.rolling(window=lookback).max()
    last_price, last_indicator = price_series.iloc[-1], indicator_series.iloc[-1]
    if last_price <= price_lows.iloc[-2] and last_indicator > indicator_lows.iloc[-2]: return 'buy'
    if last_price >= price_highs.iloc[-2] and last_indicator < indicator_highs.iloc[-2]: return 'sell'
    return 'none'

def calculate_pivot_points(df_daily):
    if df_daily is None or len(df_daily) < 1: return None
    prev_day = df_daily.iloc[-1]
    high, low, close = prev_day['high'], prev_day['low'], prev_day['close']
    pivot = (high + low + close) / 3
    return {
        'r2': pivot + (high - low), 'r1': (2 * pivot) - low,
        'pivot': pivot,
        's1': (2 * pivot) - high, 's2': pivot - (high - low)
    }

def calculate_fibonacci_retracement(df, lookback):
    recent_data = df.tail(lookback)
    high, low = recent_data['high'].max(), recent_data['low'].min()
    if high == low: return None
    diff = high - low
    return {
        'level236': high - diff * 0.236, 'level382': high - diff * 0.382,
        'level500': high - diff * 0.5,   'level618': high - diff * 0.618,
    }

def analyze_pair(pair_str, timeframe_key):
    tf_config = timeframeConfigs.get(timeframe_key)
    if not tf_config: raise ValueError(f"Invalid timeframe key: {timeframe_key}")

    asset_class = get_asset_class(pair_str)
    merged_config = get_merged_config(asset_class)
    params, weights = merged_config['params'], merged_config['weights']
    
    pair_info = parse_pair(pair_str)
    base_symbol, quote_symbol = pair_info['base'], pair_info['quote']

    df = fetch_data(base_symbol, quote_symbol, tf_config['endpoint'], tf_config['limit'] + 100, tf_config['aggregate'])
    if df is None or len(df) < tf_config['limit']: return None
        
    df.ta.ema(length=params['emaShort'], append=True, col_names=(f"EMA_S",))
    df.ta.ema(length=params['emaLong'], append=True, col_names=(f"EMA_L",))
    df.ta.rsi(length=params['rsiPeriod'], append=True, col_names=(f"RSI",))
    df.ta.bbands(length=params['bbPeriod'], std=params['bbStdDev'], append=True, col_names=(f"BBL", f"BBM", f"BBU", f"BBB", f"BBP"))
    df.ta.macd(fast=params['emaShort'], slow=params['emaLong'], signal=params['macdSignal'], append=True, col_names=(f"MACD", f"MACDh", f"MACDs"))
    df.ta.adx(length=params['adxPeriod'], append=True, col_names=(f"ADX", f"DMP", f"DMN"))
    df.ta.atr(length=params['atrPeriod'], append=True, col_names=(f"ATR",))
    df.ta.obv(append=True, col_names=(f"OBV",))
    df.ta.kc(length=params['kc_period'], scalar=params['kc_mult'], append=True, col_names=(f"KCL", f"KCM", f"KCU"))
    candlestick_patterns = df.ta.cdl_pattern(name="all")
    df = pd.concat([df, candlestick_patterns], axis=1)

    pivot_levels, fib_levels = None, None
    if asset_class in ['forex', 'gold']:
        df_daily = fetch_data(base_symbol, quote_symbol, 'histoday', 2, 1)
        pivot_levels = calculate_pivot_points(df_daily)
        fib_levels = calculate_fibonacci_retracement(df, params['fib_lookback'])

    last, prev = df.iloc[-1], df.iloc[-2]
    score, key_reasons = 0.0, []
    
    is_trending = last['ADX'] > params['adxThreshold']
    market_regime = 'トレンド' if is_trending else 'レンジ'
    key_reasons.append(f"<span class='text-purple-400'>市場環境: {market_regime} (ADX: {last['ADX']:.1f})</span>")

    dynamic_weights = weights.copy()
    if is_trending:
        dynamic_weights.update({'ma': weights['ma'] * 1.5, 'macd': weights['macd'] * 1.5, 'rsi': weights['rsi'] * 0.5, 'bb': weights['bb'] * 0.7})
    else:
        dynamic_weights.update({'ma': weights['ma'] * 0.5, 'macd': weights['macd'] * 0.5, 'rsi': weights['rsi'] * 1.5, 'bb': weights['bb'] * 1.5})
    
    if prev['EMA_S'] < prev['EMA_L'] and last['EMA_S'] > last['EMA_L']: score += dynamic_weights['ma']; key_reasons.append(f"▲ EMAゴールデンクロス")
    elif prev['EMA_S'] > prev['EMA_L'] and last['EMA_S'] < last['EMA_L']: score -= dynamic_weights['ma']; key_reasons.append(f"▼ EMAデッドクロス")

    last_candle_patterns = df.iloc[-1].filter(like='CDL_')
    bullish_patterns = last_candle_patterns[last_candle_patterns == 100]
    if not bullish_patterns.empty: score += dynamic_weights['candlestick']; key_reasons.append(f"<span class='text-teal-400'>📈 強気パターン: {bullish_patterns.index[0][4:]}</span>")
    bearish_patterns = last_candle_patterns[last_candle_patterns == -100]
    if not bearish_patterns.empty: score -= dynamic_weights['candlestick']; key_reasons.append(f"<span class='text-pink-400'>📉 弱気パターン: {bearish_patterns.index[0][4:]}</span>")
    
    if pivot_levels:
        price = last['close']
        if price > prev['close'] and prev['close'] < pivot_levels['s1'] and price > pivot_levels['s1']: score += dynamic_weights['pivot']; key_reasons.append("🔑 ピボットS1で反発")
        if price < prev['close'] and prev['close'] > pivot_levels['r1'] and price < pivot_levels['r1']: score -= dynamic_weights['pivot']; key_reasons.append("🔑 ピボットR1で反落")

    if fib_levels:
        price = last['close']
        if is_trending:
             if price > fib_levels['level618'] and prev['close'] < fib_levels['level618']: score += dynamic_weights['fibonacci']; key_reasons.append("🎯 フィボナッチ61.8%押し目")
             if price < fib_levels['level382'] and prev['close'] > fib_levels['level382']: score -= dynamic_weights['fibonacci']; key_reasons.append("🎯 フィボナッチ38.2%戻り")

    final_signal = 'hold'
    if score >= params['signalThreshold']: final_signal = 'buy'
    elif score <= -params['signalThreshold']: final_signal = 'sell'
        
    current_price = last['close']
    last_atr = last['ATR'] if 'ATR' in last and pd.notna(last['ATR']) else current_price * 0.01
    
    rr_ratio = 2.0 if is_trending else 1.5
    stop_loss = current_price - (last_atr * 1.5) if final_signal == 'buy' else current_price + (last_atr * 1.5)
    take_profit = current_price + (last_atr * 1.5 * rr_ratio) if final_signal == 'buy' else current_price - (last_atr * 1.5 * rr_ratio)

    return {"pair": pair_str, "signal": final_signal, "score": round(score, 2), "keyReasons": key_reasons if len(key_reasons) > 1 else ["<span class='text-yellow-400'>― 中立:</span> 明確なシグナルなし"], "currentPrice": current_price, "stopLoss": stop_loss, "takeProfit": take_profit}

@app.route('/analyze', methods=['GET'])
def analyze():
    pair, timeframe = request.args.get('pair'), request.args.get('timeframe')
    if not pair or not timeframe: return jsonify({"error": "Missing 'pair' or 'timeframe' parameter"}), 400
    try:
        result = analyze_pair(pair, timeframe)
        if result is None: return jsonify({"error": f"Could not analyze {pair} on {timeframe}. Data might be insufficient."}), 500
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred during analysis."}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

