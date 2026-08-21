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

    # ------------------------------------------
    # BASELINE
    # ------------------------------------------

    base_quantity = remaining / intervals_left

    # ------------------------------------------
    # MARKET CONDITIONS
    # ------------------------------------------

    liquidity_ratio = liquidity / max(volume, 1)

    spread_score = 1 / (1 + spread * 20)

    volatility_score = 1 / (1 + volatility)

    market_quality = (
        0.5 * liquidity_ratio
        + 0.3 * spread_score
        + 0.2 * volatility_score
    )

    # ------------------------------------------
    # PARTICIPATION LIMIT
    # ------------------------------------------

    # Don't consume too much available liquidity.
    #
    # This is the key difference from our
    # previous strategy.

    max_participation = 0.10

    liquidity_limit = liquidity * max_participation

    # ------------------------------------------
    # ADAPTIVE SIZE
    # ------------------------------------------

    # Market quality adjusts our normal pace,
    # but only modestly.

    multiplier = 0.85 + 0.30 * market_quality

    desired_quantity = base_quantity * multiplier

    # Never exceed our liquidity participation limit.
    quantity = min(
        desired_quantity,
        liquidity_limit
    )

    # ------------------------------------------
    # DEADLINE PROTECTION
    # ------------------------------------------

    # Near the end, prioritize completion.

    if intervals_left <= 5:
        quantity = max(
            quantity,
            remaining / intervals_left
        )

    # Never trade more than remaining.
    quantity = min(quantity, remaining)

    return quantity