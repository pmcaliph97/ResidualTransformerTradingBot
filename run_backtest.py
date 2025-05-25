# run_backtest.py
# Script to backtest a stock based on user input

from trading_bot import ResidualTransformerBot

def main():
    ticker = input("Enter stock ticker for backtest: ").upper()
    bot = ResidualTransformerBot(ticker)
    bot.fetch_data()
    bot.train_model()
    bot.generate_signals()
    bot.evaluate_model()
    bot.backtest_strategy()

if __name__ == "__main__":
    main()
