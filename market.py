import numpy as np
import pandas as pd


def generate_market_data(n=100, seed=42):

    np.random.seed(seed)

    # Starting stock price
    starting_price = 100

    # Random price movement
    price_changes = np.random.normal(0, 0.2, n)

    prices = starting_price + np.cumsum(price_changes)

    # Market volatility
    volatility = np.random.uniform(0.2, 1.0, n)

    # Trading volume
    volume = np.random.randint(10000, 100000, n)

    # Available liquidity
    liquidity = volume * np.random.uniform(0.2, 0.8, n)

    # Bid-ask spread
    spread = np.random.uniform(0.02, 0.15, n)

    bid = prices - spread / 2
    ask = prices + spread / 2

    market = pd.DataFrame({
        "price": prices,
        "bid": bid,
        "ask": ask,
        "spread": spread,
        "volume": volume,
        "liquidity": liquidity,
        "volatility": volatility
    })

    return market