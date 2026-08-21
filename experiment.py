from market import generate_market_data
from simulator import run_simulation, run_twap_simulation


TOTAL_QUANTITY = 100000
NUM_EXPERIMENTS = 100

results = []


for seed in range(NUM_EXPERIMENTS):

    # Generate a different market
    market = generate_market_data(seed=seed)

    # Run adaptive strategy
    adaptive = run_simulation(
        market,
        TOTAL_QUANTITY
    )

    # Run TWAP
    twap = run_twap_simulation(
        market,
        TOTAL_QUANTITY
    )

    adaptive_cost = adaptive["total_cost"]
    twap_cost = twap["total_cost"]

    improvement = (
        (twap_cost - adaptive_cost)
        / twap_cost
    ) * 100

    results.append(improvement)


# ==========================================
# RESULTS
# ==========================================

average = sum(results) / len(results)

wins = sum(
    1 for x in results
    if x > 0
)

win_rate = wins / NUM_EXPERIMENTS * 100

best = max(results)
worst = min(results)


print()
print("======================================")
print("       STRATEGY ROBUSTNESS TEST")
print("======================================")

print(f"Experiments:       {NUM_EXPERIMENTS}")
print(f"Average improvement: {average:.4f}%")
print(f"Win rate:            {win_rate:.2f}%")
print(f"Best case:           {best:.4f}%")
print(f"Worst case:          {worst:.4f}%")