# run_paper_trade.py
# Script to perform a paper trade on a stock based on user input

from trading_bot import ResidualTransformerBot
import alpaca_trade_api as tradeapi
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL

def main():
    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL)
    ticker = input("Enter stock ticker to paper trade: ").upper()
    bot = ResidualTransformerBot(ticker)
    bot.fetch_data()
    bot.train_model()
    bot.generate_signals()

    last_signal = bot.signals.iloc[-1]["Signal"]
    price = bot.df.iloc[-1]["Close_KF"]
    cash = float(api.get_account().cash)
    allocation = cash * 0.2
    shares = int(allocation // price)

    if last_signal in ["BUY", "SELL"] and shares > 0:
        bot.place_paper_trade(last_signal, ticker, price, shares)
    else:
        print(f"No trade executed for {ticker} (Signal: {last_signal}, Shares: {shares})")

if __name__ == "__main__":
    main()
