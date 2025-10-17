import os
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from flask import Flask, render_template, request, make_response
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# --- Basic Setup ---
app = Flask(__name__)
app.config['LSTM_MODELS_FOLDER'] = 'saved_models'
app.config['RF_MODELS_FOLDER'] = 'saved_models_rf'

# --- Global Variables ---
LOOKBACK_LSTM = 60
LOOKBACK_RF = 10
FEATURES_LSTM = ['Open', 'High', 'Low', 'Close', 'Volume', 'Log_Return']
CURRENCY_MAP = {"USD": "$", "EUR": "€", "JPY": "¥", "GBP": "£", "INR": "₹"}


# --- Data Fetching & Feature Engineering ---
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="15y")
        if df.empty: return None, None
        info = stock.info
        currency_code = info.get('currency', 'USD');
        currency_symbol = CURRENCY_MAP.get(currency_code, currency_code + " ")
        df.index = df.index.tz_localize(None)
        return df, currency_symbol
    except Exception:
        return None, None


def calculate_technical_indicators(df):
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['Volatility'] = df['Close'].pct_change().rolling(window=30).std() * np.sqrt(252)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    df.dropna(inplace=True)
    return df


# --- All Model-Specific Functions (LSTM & RF) remain the same ---
def preprocess_for_lstm(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[FEATURES_LSTM].values)
    training_data_len = int(np.ceil(len(scaled_data) * 0.8))
    return scaler, scaled_data, training_data_len


def train_new_lstm_model(scaled_data, training_data_len):
    train_data = scaled_data[0:training_data_len]
    X_train, y_train = [], []
    for i in range(LOOKBACK_LSTM, len(train_data)):
        X_train.append(train_data[i - LOOKBACK_LSTM:i]);
        y_train.append(train_data[i, -1])
    X_train, y_train = np.array(X_train), np.array(y_train)
    model = Sequential(
        [Input(shape=(X_train.shape[1], X_train.shape[2])), LSTM(128, return_sequences=True), Dropout(0.2),
         LSTM(64, return_sequences=False), Dropout(0.2), Dense(25), Dense(1)])
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.1, callbacks=[early_stopping], verbose=0)
    return model


def predict_future_lstm(model, scaler, last_60_days_scaled, last_known_price, n_days):
    future_predictions = [];
    current_sequence = last_60_days_scaled.reshape(1, LOOKBACK_LSTM, len(FEATURES_LSTM));
    current_price = last_known_price
    for _ in range(n_days):
        next_log_return_scaled = model.predict(current_sequence, verbose=0)[0][0]
        dummy = np.zeros((1, len(FEATURES_LSTM)));
        dummy[0, -1] = next_log_return_scaled
        next_log_return = scaler.inverse_transform(dummy)[0, -1]
        next_price = current_price * np.exp(next_log_return)
        future_predictions.append(next_price);
        current_price = next_price
        new_row = current_sequence[0, -1, :].copy();
        new_row[-1] = next_log_return_scaled
        current_sequence = np.append(current_sequence[:, 1:, :], new_row.reshape(1, 1, len(FEATURES_LSTM)), axis=1)
    return future_predictions


def preprocess_for_rf(data, target_col='Log_Return', n_lags=LOOKBACK_RF):
    df_rf = pd.DataFrame(index=data.index)
    for i in range(1, n_lags + 1):
        df_rf[f'lag_{i}'] = data[target_col].shift(i)
    df_rf['target'] = data[target_col]
    df_rf.dropna(inplace=True)
    X = df_rf.drop('target', axis=1)
    y = df_rf['target']
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, _ = y[:split_idx], y[split_idx:]
    return X_train, y_train, X_test


def train_new_rf_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
    model.fit(X_train, y_train)
    return model


def predict_future_rf(model, last_lags, last_known_price, n_days):
    future_predictions = [];
    current_price = last_known_price;
    recent_lags = list(last_lags)
    for _ in range(n_days):
        prediction_features = np.array(recent_lags).reshape(1, -1)
        next_log_return = model.predict(prediction_features)[0]
        next_price = current_price * np.exp(next_log_return)
        future_predictions.append(next_price);
        current_price = next_price
        recent_lags = recent_lags[1:] + [next_log_return]
    return future_predictions


# --- Plotting Function (Shared) ---
def create_plots(ticker, data, valid_df, future_dates, future_preds):
    """
    FIX: Explicitly convert all pandas Series/Index objects to lists
    before passing them to Plotly. This guarantees a clean JSON format.
    """
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=data.index.tolist(), y=data['Close'].tolist(), name='Historical Close',
                                   line=dict(color='#636EFA')))
    fig_price.add_trace(go.Scatter(x=data.index.tolist(), y=data['SMA50'].tolist(), name='50-Day SMA',
                                   line=dict(color='#FFA15A', width=1.5, dash='dash')))
    fig_price.add_trace(go.Scatter(x=data.index.tolist(), y=data['SMA200'].tolist(), name='200-Day SMA',
                                   line=dict(color='#00CC96', width=1.5, dash='dash')))
    fig_price.add_trace(
        go.Scatter(x=valid_df.index.tolist(), y=valid_df['Predictions'].tolist(), name='Predicted Price (Test)',
                   line=dict(color='#EF553B')))
    fig_price.add_trace(go.Scatter(x=future_dates.tolist(), y=future_preds, name='Future Forecast',
                                   line=dict(color='#AB63FA', dash='dot')))
    fig_price.update_layout(title_text=f'{ticker} Price, Moving Averages & Prediction', template='plotly_white',
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data.index.tolist(), y=data['RSI'].tolist(), name='RSI'))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought");
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig_rsi.update_layout(title_text='Relative Strength Index (RSI)', template='plotly_white')

    fig_vol = make_subplots(specs=[[{"secondary_y": True}]])
    fig_vol.add_trace(
        go.Bar(x=data.index.tolist(), y=data['Volume'].tolist(), name='Volume', marker_color='rgba(99, 110, 250, 0.5)'),
        secondary_y=False)
    fig_vol.add_trace(go.Scatter(x=data.index.tolist(), y=data['Volatility'].tolist(), name='30-Day Volatility',
                                 line=dict(color='#EF553B')), secondary_y=True)
    fig_vol.update_layout(title_text='Trading Volume & Annualized Volatility', template='plotly_white')

    graph_price_json = json.dumps(fig_price, cls=plotly.utils.PlotlyJSONEncoder)
    graph_rsi_json = json.dumps(fig_rsi, cls=plotly.utils.PlotlyJSONEncoder)
    graph_vol_json = json.dumps(fig_vol, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_price_json, graph_rsi_json, graph_vol_json


@app.route('/')
def index():
    # Get separate lists for each model type
    lstm_models = sorted(
        [f.replace('lstm_model_', '').replace('.keras', '') for f in os.listdir(app.config['LSTM_MODELS_FOLDER']) if
         f.endswith('.keras')])
    rf_models = sorted(
        [f.replace('rf_model_', '').replace('.joblib', '') for f in os.listdir(app.config['RF_MODELS_FOLDER']) if
         f.endswith('.joblib')])

    # Pass both lists to the template
    response = make_response(render_template('index.html', lstm_models=lstm_models, rf_models=rf_models))

    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/predict', methods=['POST'])
def predict():
    model_type = request.form.get('model_type')
    ticker_select = request.form.get('ticker_select');
    ticker_input = request.form.get('ticker_input', '').upper();
    predict_days = int(request.form.get('predict_days', 30))
    ticker = ticker_input if ticker_input else ticker_select
    ticker_for_yfinance = ticker.replace('.BSE', '.BO') if '.BSE' in ticker else ticker
    raw_data, currency_symbol = fetch_stock_data(ticker_for_yfinance)
    if raw_data is None: return render_template('results.html',
                                                error=f"Could not fetch data for ticker: {ticker_for_yfinance}.")

    data_with_indicators = calculate_technical_indicators(raw_data.copy())

    if model_type == 'lstm':
        model_path = os.path.join(app.config['LSTM_MODELS_FOLDER'], f'lstm_model_{ticker}.keras')
        scaler, scaled_data, training_data_len = preprocess_for_lstm(data_with_indicators)
        force_retrain = False
        if os.path.exists(model_path) and not ticker_input:
            try:
                temp_model = load_model(model_path)
                if temp_model.input_shape[-1] != len(FEATURES_LSTM): force_retrain = True
            except:
                force_retrain = True
        if os.path.exists(model_path) and not ticker_input and not force_retrain:
            model = load_model(model_path)
        else:
            model = train_new_lstm_model(scaled_data, training_data_len); model.save(model_path)

        test_data_scaled = scaled_data[training_data_len - LOOKBACK_LSTM:]
        X_test = np.array([test_data_scaled[i - LOOKBACK_LSTM:i] for i in range(LOOKBACK_LSTM, len(test_data_scaled))])
        predicted_log_returns_scaled = model.predict(X_test, verbose=0)
        dummy = np.zeros((len(predicted_log_returns_scaled), len(FEATURES_LSTM)));
        dummy[:, -1] = predicted_log_returns_scaled.flatten()
        predicted_log_returns = scaler.inverse_transform(dummy)[:, -1]

        valid_df = data_with_indicators.iloc[training_data_len:].copy()
        y_test = valid_df['Close'].values
        last_prices_in_test = data_with_indicators['Close'].iloc[training_data_len - 1:-1].values
        min_len = min(len(last_prices_in_test), len(predicted_log_returns))
        predictions = last_prices_in_test[:min_len] * np.exp(predicted_log_returns[:min_len])
        y_test = y_test[:min_len]
        valid_df = valid_df.iloc[:min_len];
        valid_df['Predictions'] = predictions

        last_60_days_scaled = scaled_data[-LOOKBACK_LSTM:]
        future_preds = predict_future_lstm(model, scaler, last_60_days_scaled, data_with_indicators['Close'].iloc[-1],
                                           predict_days)

    elif model_type == 'rf':
        model_path = os.path.join(app.config['RF_MODELS_FOLDER'], f'rf_model_{ticker}.joblib')
        X_train, y_train, X_test = preprocess_for_rf(data_with_indicators)
        if os.path.exists(model_path) and not ticker_input:
            model = joblib.load(model_path)
        else:
            model = train_new_rf_model(X_train, y_train); joblib.dump(model, model_path)

        predicted_log_returns = model.predict(X_test)
        valid_df = data_with_indicators.loc[X_test.index].copy()
        y_test = valid_df['Close'].values
        last_prices_in_test = data_with_indicators['Close'].shift(1).loc[X_test.index].values
        predictions = last_prices_in_test * np.exp(predicted_log_returns)
        valid_df['Predictions'] = predictions

        last_lags = data_with_indicators['Log_Return'].iloc[-LOOKBACK_RF:].values
        future_preds = predict_future_rf(model, last_lags, data_with_indicators['Close'].iloc[-1], predict_days)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    last_date = data_with_indicators.index[-1]
    future_dates = pd.to_datetime([last_date + pd.DateOffset(days=i) for i in range(1, predict_days + 1)])

    final_predicted_rate = future_preds[-1] if future_preds else None
    final_predicted_date = future_dates[-1].strftime('%B %d, %Y') if not future_dates.empty else "N/A"

    graph_price_json, graph_rsi_json, graph_vol_json = create_plots(ticker, data_with_indicators, valid_df,
                                                                    future_dates, future_preds)

    return render_template('results.html',
                           ticker=ticker, currency=currency_symbol, model_type=model_type,
                           current_rate=f"{data_with_indicators['Close'].iloc[-1]:,.2f}",
                           predicted_rate=f"{future_preds[0]:,.2f}" if future_preds else "N/A",
                           volatility=f"{data_with_indicators['Volatility'].iloc[-1]:.2%}",
                           r2=f"{r2:.4f}", rmse=f"{currency_symbol}{rmse:,.2f}", mae=f"{currency_symbol}{mae:,.2f}",
                           predict_days=predict_days,
                           final_predicted_rate=f"{final_predicted_rate:,.2f}" if final_predicted_rate else "N/A",
                           final_predicted_date=final_predicted_date,
                           graph_price_json=graph_price_json, graph_rsi_json=graph_rsi_json,
                           graph_vol_json=graph_vol_json
                           )


if __name__ == '__main__':
    app.run(debug=True)

