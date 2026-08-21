def twap_strategy(total_quantity, current_time, total_time, remaining):

    intervals_left = total_time - current_time

    if intervals_left <= 0:
        return remaining

    quantity = remaining / intervals_left

    return quantity
def adaptive_strategy(
    remaining,
    current_time,
    total_time,
    spread,
    volume,
    liquidity,
    volatility
):

    if remaining <= 0:
        return 0

    intervals_left = total_time - current_time

    if intervals_left <= 1:
        return remaining

    # Amount we need to trade on average
    base_quantity = remaining / intervals_left

    # More liquidity = we can trade more
    liquidity_ratio = liquidity / max(volume, 1)

    # Wider spread = more expensive to trade
    spread_penalty = 1 / (1 + spread * 10)

    # Higher volatility = more risky to trade
    volatility_penalty = 1 / (1 + volatility)

    # Deadline pressure
    urgency = 1 + (1 / intervals_left)

    quantity = (
        base_quantity
        * (0.5 + liquidity_ratio)
        * spread_penalty
        * volatility_penalty
        * urgency
    )

    quantity = min(quantity, remaining)

    return quantity