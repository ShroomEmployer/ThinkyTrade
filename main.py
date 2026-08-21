from market import generate_market_data
from simulator import run_simulation


# Generate market
market = generate_market_data()

# Order we want to execute
total_quantity = 100000


# Run simulation
result = run_simulation(
    market,
    total_quantity
)


print("\nFINTECH EXECUTION ENGINE")
print("========================")

print(f"Original order:    {total_quantity:.2f}")
print(f"Executed:          {result['total_executed']:.2f}")
print(f"Remaining:         {result['remaining']:.2f}")
print(f"Total cost:        ₹{result['total_cost']:,.2f}")


print("\nFirst 5 trades")
print("--------------")


for trade in result["trades"][:5]:

    print(
        f"Time {trade['time']:2d} | "
        f"Quantity {trade['quantity']:8.2f} | "
        f"Price ₹{trade['price']:.2f} | "
        f"Remaining {trade['remaining']:10.2f}"
    )