from strategy import adaptive_strategy, twap_strategy
def calculate_execution_price(row, quantity):

    base_price = row["ask"]

    available_liquidity = max(row["liquidity"], 1)

    # How large is our order compared
    # with available liquidity?
    participation = quantity / available_liquidity

    # Market impact increases as we consume
    # more of the available liquidity.
    impact = 0.05 * (participation ** 0.5)

    execution_price = base_price * (1 + impact)

    return execution_price


def run_simulation(market, total_quantity):

    remaining = total_quantity
    total_cost = 0
    total_executed = 0

    trades = []

    total_time = len(market)

    for t in range(total_time):

        # Stop if order is already completed
        if remaining <= 0:
            break

        row = market.iloc[t]

        # Ask our strategy how much to trade
        quantity = adaptive_strategy(
            remaining=remaining,
            current_time=t,
            total_time=total_time,
            spread=row["spread"],
            volume=row["volume"],
            liquidity=row["liquidity"],
            volatility=row["volatility"]
        )

        # Never trade more than we actually need
        quantity = min(quantity, remaining)

        # We are BUYING, so we pay the ASK price
        execution_price = calculate_execution_price(
        row,
        quantity
        )

        # Money spent on this trade
        cost = quantity * execution_price

        # Update totals
        remaining -= quantity
        total_executed += quantity
        total_cost += cost

        trades.append({
            "time": t,
            "quantity": quantity,
            "price": execution_price,
            "cost": cost,
            "remaining": remaining
        })

    return {
        "total_executed": total_executed,
        "remaining": remaining,
        "total_cost": total_cost,
        "trades": trades
    }
def run_twap_simulation(market, total_quantity):

    remaining = total_quantity
    total_cost = 0
    total_executed = 0

    trades = []

    total_time = len(market)

    for t in range(total_time):

        if remaining <= 0:
            break

        row = market.iloc[t]

        quantity = twap_strategy(
            total_quantity=total_quantity,
            current_time=t,
            total_time=total_time,
            remaining=remaining
        )

        quantity = min(quantity, remaining)

        execution_price = row["ask"]

        cost = quantity * execution_price

        remaining -= quantity
        total_executed += quantity
        total_cost += cost

        trades.append({
            "time": t,
            "quantity": quantity,
            "price": execution_price,
            "cost": cost,
            "remaining": remaining
        })

    return {
        "total_executed": total_executed,
        "remaining": remaining,
        "total_cost": total_cost,
        "trades": trades
    }