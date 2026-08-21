def evaluate_execution(
    total_quantity,
    arrival_price,
    total_cost,
    total_executed
):

    # What the order would have cost
    # if we could buy everything at the initial price.
    benchmark_cost = total_quantity * arrival_price

    # Extra money spent compared to benchmark.
    shortfall = total_cost - benchmark_cost

    # Average price we actually paid.
    if total_executed > 0:
        average_price = total_cost / total_executed
    else:
        average_price = 0

    # Percentage of the order completed.
    completion = total_executed / total_quantity

    return {
        "benchmark_cost": benchmark_cost,
        "actual_cost": total_cost,
        "shortfall": shortfall,
        "average_price": average_price,
        "completion": completion
    }