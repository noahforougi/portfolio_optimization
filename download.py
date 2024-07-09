import yfinance as yf

import config


def pull_prices():
    data = yf.download(config.TICKERS, start=config.START_DATE, end=config.END_DATE)[
        "Adj Close"
    ].reset_index()
    return data
