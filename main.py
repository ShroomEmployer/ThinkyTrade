from market import generate_market_data
from simulator import run_simulation, run_twap_simulation
from evaluation import evaluate_execution


TOTAL_QUANTITY = 100000


# ==========================================
# GENERATE ONE MARKET
# ==========================================

market = generate_market_data()

arrival_price = market.iloc[0]["price"]


# ==========================================
# RUN ADAPTIVE STRATEGY
# ==========================================

adaptive_result = run_simulation(
    market,
    TOTAL_QUANTITY
)

adaptive_evaluation = evaluate_execution(
    total_quantity=TOTAL_QUANTITY,
    arrival_price=arrival_price,
    total_cost=adaptive_result["total_cost"],
    total_executed=adaptive_result["total_executed"]
)


# ==========================================
# RUN TWAP
# ==========================================

twap_result = run_twap_simulation(
    market,
    TOTAL_QUANTITY
)

twap_evaluation = evaluate_execution(
    total_quantity=TOTAL_QUANTITY,
    arrival_price=arrival_price,
    total_cost=twap_result["total_cost"],
    total_executed=twap_result["total_executed"]
)


# ==========================================
# COMPARE
# ==========================================

adaptive_cost = adaptive_evaluation["actual_cost"]
twap_cost = twap_evaluation["actual_cost"]

cost_difference = twap_cost - adaptive_cost

improvement = (
    cost_difference / twap_cost
) * 100


print()
print("======================================")
print("      FINTECH EXECUTION ENGINE")
print("======================================")

print(f"Order size:       {TOTAL_QUANTITY:,.0f}")
print(f"Arrival price:    ₹{arrival_price:.4f}")

print()
print("STRATEGY COMPARISON")
print("--------------------------------------")

print(
    f"TWAP cost:        "
    f"₹{twap_cost:,.2f}"
)

print(
    f"Adaptive cost:    "
    f"₹{adaptive_cost:,.2f}"
)

print(
    f"Difference:       "
    f"₹{cost_difference:,.2f}"
)

print(
    f"Improvement:      "
    f"{improvement:.4f}%"
)

print()
print("COMPLETION")
print("--------------------------------------")

print(
    f"TWAP:             "
    f"{twap_evaluation['completion'] * 100:.2f}%"
)

print(
    f"Adaptive:         "
    f"{adaptive_evaluation['completion'] * 100:.2f}%"
)