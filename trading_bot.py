import torch
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from nltk.sentiment import SentimentIntensityAnalyzer
from ta.trend import EMAIndicator
from config import *
from model import TransformerModel
from data_loader import fetch_data
import alpaca_trade_api as tradeapi

class ResidualTransformerBot:
    def __init__(self, stock_symbol):
        self.stock_symbol = stock_symbol.upper()
        self.start_date = (pd.Timestamp.today() - pd.DateOffset(years=4)).date().isoformat()
        self.end_date = pd.Timestamp.today().date().isoformat()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sia = SentimentIntensityAnalyzer()

    def fetch_data(self):
        self.df = fetch_data(self.stock_symbol, self.start_date, self.end_date)
        print(f"✅ Fetched {len(self.df)} rows for {self.stock_symbol}")

    def train_model(self, window=30, epochs=200, lr=0.0005):
        df = self.df.copy()
        features = ["Close_KF", "EMA20", "RSI"]
        target = (df["Target"] - df["Previous_Close_KF"]).values.reshape(-1, 1)

        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        X_scaled = feature_scaler.fit_transform(df[features])
        y_scaled = target_scaler.fit_transform(target)

        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

        split = int(len(X_scaled) * 0.8)
        X_train = np.array([X_scaled[i:i + window] for i in range(split - window)])
        y_train = np.array([y_scaled[i + window] for i in range(split - window)])
        X_test = np.array([X_scaled[i:i + window] for i in range(split, len(X_scaled) - 1)])
        y_test = np.array([y_scaled[i + window] for i in range(split, len(y_scaled) - 1)])
        prev_close_test = df["Previous_Close_KF"].values[split + window:]

        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        model = TransformerModel(input_dim=X_train.shape[2]).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = torch.nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(X_train).squeeze()
            loss = loss_fn(out, y_train.squeeze())
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

        model.eval()
        with torch.no_grad():
            pred_residuals_scaled = model(X_test).squeeze().cpu().numpy()

        pred_residuals = self.target_scaler.inverse_transform(pred_residuals_scaled.reshape(-1, 1)).flatten()
        y_true_residuals = self.target_scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).flatten()

        self.y_true = prev_close_test + y_true_residuals
        self.y_pred = prev_close_test + pred_residuals
        self.model = model
        self.pred_df = self.df.iloc[-len(self.y_pred):].copy()
        self.pred_df["Predicted"] = self.y_pred
        self.pred_df["Actual"] = self.y_true

    def generate_signals(self):
        df = self.pred_df.copy()
        df["Predicted_Next"] = df["Predicted"].shift(-1)
        df["Pct_Change"] = (df["Predicted_Next"] - df["Predicted"]) / df["Predicted"]
        df["Signal"] = "HOLD"
        df.loc[(df["Predicted_Next"] < df["Predicted"]) & (df["Pct_Change"] < -THRESHOLD), "Signal"] = "SELL"
        df.loc[(df["Predicted_Next"] > df["Predicted"]) & (df["Pct_Change"] > THRESHOLD), "Signal"] = "BUY"
        self.signals = df[["date", "Predicted", "Predicted_Next", "Pct_Change", "Signal"]].dropna()
        print(self.signals.tail())

    def evaluate_model(self):
        y_true, y_pred = self.y_true, self.y_pred
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f} | MAPE: {mape:.2f}%")

        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label="Actual")
        plt.plot(y_pred, label="Predicted")
        plt.title(f"{self.stock_symbol} Prediction vs Actual")
        plt.legend()
        plt.grid()
        plt.show()


    def backtest_strategy(self, initial_cash=100000, risk_pct=0.2, stop_loss=0.03, take_profit=0.05, fee_per_trade=1.0, cooldown_period=2):
        """Backtest the trading strategy based on predicted signals and risk management."""
        df = self.signals.copy().reset_index(drop=True)
        df["Exec_Price"] = self.pred_df["Actual"].values[-len(df):]
        df["EMA10"] = EMAIndicator(close=self.pred_df["Close_KF"]).ema_indicator()[-len(df):].values
        df["EMA20"] = EMAIndicator(close=self.pred_df["Close_KF"]).ema_indicator()[-len(df):].values

        position = None
        entry_price = 0
        cash = initial_cash
        shares = 0
        cash_curve = []
        trade_log = []
        trade_returns = []
        last_trade_index = -cooldown_period

        for i in range(len(df) - 1):
            signal = df.loc[i, "Signal"]
            price = df.loc[i + 1, "Exec_Price"]
            date = df.loc[i + 1, "date"]
            ema10 = df.loc[i + 1, "EMA10"]
            ema20 = df.loc[i + 1, "EMA20"]

            if i - last_trade_index < cooldown_period:
                cash_curve.append(cash)
                continue

            if position == "LONG":
                current_return = (price - entry_price) / entry_price
                trend_broken = ema10 < ema20
                if trend_broken or current_return <= -stop_loss or current_return >= take_profit:
                    cash += shares * price - fee_per_trade
                    trade_returns.append(current_return)
                    trade_log.append((date, "Exit LONG", price, shares, current_return))
                    shares = 0
                    position = None
                    last_trade_index = i

            elif position == "SHORT":
                current_return = (entry_price - price) / entry_price
                trend_broken = ema10 > ema20
                if trend_broken or current_return <= -stop_loss or current_return >= take_profit:
                    cash += shares * (2 * entry_price - price) - fee_per_trade
                    trade_returns.append(current_return)
                    trade_log.append((date, "Exit SHORT", price, shares, current_return))
                    shares = 0
                    position = None
                    last_trade_index = i

            if signal == "BUY" and position is None:
                allocation = cash * risk_pct
                shares = allocation // price
                entry_price = price
                cash -= shares * price + fee_per_trade
                position = "LONG"
                trade_log.append((date, "Enter LONG", price, shares))
                last_trade_index = i

            elif signal == "SELL" and position is None:
                allocation = cash * risk_pct
                shares = allocation // price
                entry_price = price
                cash += shares * price - fee_per_trade
                position = "SHORT"
                trade_log.append((date, "Enter SHORT", price, shares))
                last_trade_index = i

            cash_curve.append(cash)

        df_bt = pd.DataFrame({"Date": df["date"].iloc[1:], "Cash": cash_curve})
        df_bt.set_index("Date", inplace=True)
        ax = df_bt["Cash"].plot(figsize=(12, 6), title="Backtest Cash Curve")

        for i, value in enumerate(df_bt["Cash"]):
            if i % (len(df_bt) // 10 + 1) == 0:
                ax.annotate(f"{int(value):,}", xy=(df_bt.index[i], value), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

        plt.grid()
        plt.ylabel("Cash ($)")
        plt.show()

        print("\nTrade Log:")
        for log in trade_log:
            print(log)

        realized_equity = cash
        total_return = (realized_equity - initial_cash) / initial_cash
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0
        win_rate = np.mean([1 if r > 0 else 0 for r in trade_returns]) if trade_returns else 0
        sharpe_ratio = (np.mean(trade_returns) / np.std(trade_returns)) * np.sqrt(252) if len(trade_returns) > 1 else 0

        print("\n--- Backtest Summary ---")
        print(f"Total Return: {total_return:.2%}")
        print(f"Average Trade Return: {avg_trade_return:.2%}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
