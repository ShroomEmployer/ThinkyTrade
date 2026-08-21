import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import json
from openai import OpenAI
from plotly.subplots import make_subplots


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Adaptive Execution Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# AI / OPENROUTER
# ============================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

ai_client = (
    OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    if OPENROUTER_API_KEY
    else None
)

AI_MODEL = "openrouter/free"


def call_execution_ai(system_prompt, user_prompt):
    if ai_client is None:
        return (
            "AI is offline. Set the OPENROUTER_API_KEY environment "
            "variable and restart Streamlit."
        )

    try:
        response = ai_client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.25,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"AI request failed: {exc}"


def execution_context(
    market, adaptive, twap, aggressive,
    order_size, scenario, adaptive_cost, twap_cost, aggressive_cost
):
    latest = market.iloc[-1]
    quality = market.apply(calculate_market_quality, axis=1)
    executed = adaptive["quantity"].sum()

    return {
        "regime": scenario,
        "order_size": float(order_size),
        "completion_pct": float(executed / max(order_size, 1) * 100),
        "remaining": float(max(order_size - executed, 0)),
        "arrival_price": float(market.iloc[0]["price"]),
        "latest_price": float(latest["price"]),
        "latest_bid": float(latest["bid"]),
        "latest_ask": float(latest["ask"]),
        "spread": float(latest["spread"]),
        "liquidity": float(latest["liquidity"]),
        "volatility": float(latest["volatility"]),
        "latest_quality": float(calculate_market_quality(latest)),
        "min_quality": float(quality.min()),
        "max_quality": float(quality.max()),
        "avg_quality": float(quality.mean()),
        "adaptive_cost": float(adaptive_cost),
        "twap_cost": float(twap_cost),
        "aggressive_cost": float(aggressive_cost),
        "adaptive_vs_twap_pct": float(
            (twap_cost - adaptive_cost) / max(twap_cost, 1) * 100
        ),
        "avg_impact_bps": float(adaptive["impact"].mean() * 10000),
        "avg_participation_pct": float(
            adaptive["participation"].mean() * 100
        ),
        "aggressive_decisions": int(
            (adaptive["action"] == "AGGRESSIVE").sum()
        ),
        "normal_decisions": int(
            (adaptive["action"] == "NORMAL").sum()
        ),
        "cautious_decisions": int(
            (adaptive["action"] == "CAUTIOUS").sum()
        ),
    }


def ai_report(ctx):
    return call_execution_ai(
        """
You are the Execution Intelligence layer of an institutional
execution simulator. Analyze ONLY the supplied synthetic data.

Use:
## Market Assessment
## Execution Assessment
## Benchmark Analysis
## Key Risk
## AI Recommendation

Do not invent data. This is not live market data or financial advice.
""",
        f"Simulation data:\n{ctx}",
    )


def ai_decision_explanation(latest, decision, remaining, scenario):
    return call_execution_ai(
        """
Explain why the deterministic execution engine selected AGGRESSIVE,
NORMAL, or CAUTIOUS. Use only the supplied liquidity, spread,
volatility, quality, remaining order and decision data.
Keep the answer under 150 words.
""",
        f"""
Regime: {scenario}
Price: {latest['price']:.4f}
Spread: {latest['spread']:.5f}
Liquidity: {latest['liquidity']:,.0f}
Volatility: {latest['volatility']:.3f}
Quality: {decision['quality'] * 100:.1f}%
Decision: {decision['action']}
Suggested quantity: {decision['quantity']:,.0f}
Remaining: {remaining:,.0f}
""",
    )


def ai_history(adaptive):
    records = adaptive.tail(20)[
        [
            "time", "action", "quality", "liquidity",
            "spread", "volatility", "quantity",
            "participation", "impact"
        ]
    ].round(4).to_dict("records")

    return call_execution_ai(
        """
Analyze the recent decision history of an adaptive execution
algorithm. Return:
1. Observed Pattern
2. Algorithm Response
3. Potential Weakness
4. Suggested Improvement

Do not claim statistical significance.
""",
        f"Recent decisions:\n{records}",
    )


def ai_what_if(current, proposed):
    return call_execution_ai(
        """
Evaluate a synthetic execution what-if scenario. Compare the
current and proposed configurations using only supplied numbers.
Discuss speed, completion, impact, participation, cost and risk.

End with exactly one of:
VERDICT: MORE EFFICIENT
VERDICT: MORE AGGRESSIVE
VERDICT: MORE CONSERVATIVE
VERDICT: TRADE-OFF
""",
        f"CURRENT:\n{current}\n\nPROPOSED:\n{proposed}",
    )



# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1600px;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.45rem;
        font-weight: 650;
    }

    .hero {
        padding: 24px 28px;
        border: 1px solid #293241;
        border-radius: 16px;
        background:
            radial-gradient(circle at 90% 10%, rgba(79,140,255,.13), transparent 32%),
            linear-gradient(135deg, #111722, #0d1119);
        margin-bottom: 18px;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 750;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        color: #9ca8ba;
        font-size: .98rem;
    }

    .status-card {
        padding: 16px 18px;
        border-radius: 12px;
        border: 1px solid #2d3748;
        background: #121822;
        height: 100%;
    }

    .status-label {
        color: #8e9aac;
        text-transform: uppercase;
        letter-spacing: .09em;
        font-size: .72rem;
        margin-bottom: 6px;
    }

    .status-value {
        font-size: 1.15rem;
        font-weight: 700;
    }

    .positive {
        color: #43d17a;
    }

    .negative {
        color: #ff6b7a;
    }

    .neutral {
        color: #e6ebf2;
    }

    .section-label {
        color: #8e9aac;
        text-transform: uppercase;
        letter-spacing: .12em;
        font-size: .72rem;
        font-weight: 650;
        margin-bottom: 8px;
    }

    .decision-panel {
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #303a4a;
        background: #111722;
        min-height: 210px;
    }

    .decision-action {
        font-size: 1.65rem;
        font-weight: 800;
        margin: 4px 0 10px;
    }

    .decision-reason {
        color: #aab4c3;
        line-height: 1.55;
    }

    .pill {
        display: inline-block;
        padding: 4px 9px;
        border-radius: 999px;
        border: 1px solid #374151;
        background: #171e29;
        color: #cbd5e1;
        font-size: .72rem;
        margin-right: 5px;
    }

    .info-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
    }

    .info-item {
        border: 1px solid #293241;
        border-radius: 10px;
        padding: 12px;
        background: #0f151e;
    }

    .info-key {
        color: #8995a7;
        font-size: .72rem;
        text-transform: uppercase;
        letter-spacing: .07em;
    }

    .info-val {
        margin-top: 4px;
        font-size: 1rem;
        font-weight: 650;
    }

    .footer {
        text-align: center;
        color: #687386;
        padding: 18px 0 4px;
        font-size: .78rem;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MARKET GENERATOR
# ============================================================

SCENARIO_CONFIG = {
    "Normal": {
        "vol": (0.15, 0.75),
        "liq": (50_000, 150_000),
        "spread": (0.015, 0.08),
        "drift": 0.000,
    },
    "High Volatility": {
        "vol": (0.60, 1.40),
        "liq": (40_000, 130_000),
        "spread": (0.025, 0.12),
        "drift": 0.000,
    },
    "Low Liquidity": {
        "vol": (0.25, 0.90),
        "liq": (15_000, 60_000),
        "spread": (0.04, 0.15),
        "drift": 0.000,
    },
    "Stress": {
        "vol": (0.70, 1.60),
        "liq": (10_000, 50_000),
        "spread": (0.05, 0.20),
        "drift": -0.0005,
    },
}


def generate_market(seed, scenario, steps=120, start_price=100.0):
    rng = np.random.default_rng(seed)
    cfg = SCENARIO_CONFIG[scenario]

    price = float(start_price)
    rows = []

    for t in range(steps):
        volatility = rng.uniform(*cfg["vol"])
        shock = rng.normal(cfg["drift"], 0.025 * volatility)
        price = max(1.0, price * (1 + shock))

        spread = rng.uniform(*cfg["spread"])
        volume = int(rng.integers(*cfg["liq"]))
        liquidity = volume * rng.uniform(0.35, 0.90)

        bid = price - spread / 2
        ask = price + spread / 2

        rows.append(
            {
                "time": t,
                "price": price,
                "bid": bid,
                "ask": ask,
                "spread": spread,
                "volume": volume,
                "liquidity": liquidity,
                "volatility": volatility,
            }
        )

    return pd.DataFrame(rows)



# ============================================================
# AI EXECUTION AGENT
# ============================================================

def parse_ai_json(text):
    text = (text or "").strip()

    if text.startswith("```"):
        text = text.replace("```json", "", 1).replace("```", "", 1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def get_ai_execution_decision(
    market,
    row_index,
    remaining,
    total_order,
    steps_left,
    previous_decisions,
    decision_interval,
):
    """Ask the AI to choose the next execution policy."""

    if ai_client is None:
        return None

    row = market.iloc[row_index]
    start = max(0, row_index - 4)

    recent = market.iloc[start:row_index + 1][
        [
            "time",
            "price",
            "bid",
            "ask",
            "spread",
            "liquidity",
            "volume",
            "volatility",
        ]
    ].round(5).to_dict("records")

    previous = previous_decisions[-5:] if previous_decisions else []

    system_prompt = """
You are the decision-making agent inside a synthetic institutional
trade-execution simulator.

Choose HOW the next execution block should be handled.

Return ONLY valid JSON with:
{
  "action": "AGGRESSIVE | NORMAL | CAUTIOUS",
  "quantity": 1000,
  "participation": 0.08,
  "urgency": 1.0,
  "confidence": 0.80,
  "rationale": "short explanation"
}

Optimize the trade-off between completion speed, execution cost,
market impact, liquidity consumption, spread, volatility, remaining
order and time remaining.

Bounds:
- participation: 0.02 to 0.25
- urgency: 0.50 to 2.00
- confidence: 0.00 to 1.00
- quantity must be non-negative

Never invent observations. The application will enforce hard safety
limits after your decision. Do not output hidden chain-of-thought.
This is synthetic simulation data, not live trading.
"""

    user_prompt = f"""
CURRENT MARKET:
{row.to_dict()}

RECENT MARKET HISTORY:
{recent}

PREVIOUS AI DECISIONS:
{previous}

ORDER STATE:
Total order: {total_order:,.0f}
Remaining order: {remaining:,.0f}
Steps remaining: {steps_left}
Decision interval: {decision_interval} observations

Choose the next execution policy.
"""

    try:
        response = ai_client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.15,
            max_tokens=500,
        )

        decision = parse_ai_json(
            response.choices[0].message.content
        )

        action = str(
            decision.get("action", "NORMAL")
        ).upper()

        if action not in {"AGGRESSIVE", "NORMAL", "CAUTIOUS"}:
            action = "NORMAL"

        participation = float(
            np.clip(
                float(decision.get("participation", 0.10)),
                0.02,
                0.25,
            )
        )

        urgency_value = float(
            np.clip(
                float(decision.get("urgency", 1.0)),
                0.50,
                2.00,
            )
        )

        confidence = float(
            np.clip(
                float(decision.get("confidence", 0.50)),
                0.0,
                1.0,
            )
        )

        requested_quantity = max(
            float(decision.get("quantity", 0)),
            0.0,
        )

        # Hard safety boundary around the AI.
        liquidity_limit = row["liquidity"] * participation
        safe_quantity = min(
            requested_quantity,
            liquidity_limit,
            remaining,
        )

        if steps_left <= 2:
            safe_quantity = min(
                remaining,
                max(
                    safe_quantity,
                    remaining / max(steps_left, 1),
                ),
            )

        return {
            "quantity": float(max(safe_quantity, 0)),
            "action": action,
            "participation": participation,
            "urgency": urgency_value,
            "confidence": confidence,
            "rationale": str(
                decision.get(
                    "rationale",
                    "AI selected this policy from current conditions.",
                )
            ),
        }

    except Exception as exc:
        return {
            "quantity": 0.0,
            "action": "NORMAL",
            "participation": 0.10,
            "urgency": 1.0,
            "confidence": 0.0,
            "rationale": f"AI decision unavailable: {exc}",
        }


def run_ai_agent(
    market,
    total_order,
    max_participation=0.10,
    urgency=1.0,
    impact_coefficient=0.025,
    decision_interval=5,
):
    """
    AI-driven execution strategy.

    The AI makes a strategic decision every N observations.
    The deterministic executor enforces the hard limits.
    """

    remaining = float(total_order)
    total_cost = 0.0
    records = []
    previous_decisions = []
    current_policy = None

    for i, row in market.iterrows():
        if remaining <= 0:
            break

        steps_left = len(market) - i

        if (
            current_policy is None
            or i % decision_interval == 0
        ):
            current_policy = get_ai_execution_decision(
                market,
                i,
                remaining,
                total_order,
                steps_left,
                previous_decisions,
                decision_interval,
            )

            if current_policy is None:
                fallback = execution_decision(
                    row,
                    remaining,
                    total_order,
                    steps_left,
                    max_participation,
                    urgency,
                )

                current_policy = {
                    "quantity": fallback["quantity"],
                    "action": fallback["action"],
                    "participation": min(
                        max_participation,
                        0.25,
                    ),
                    "urgency": urgency,
                    "confidence": 0.0,
                    "rationale": (
                        "AI unavailable; deterministic fallback used."
                    ),
                }

            previous_decisions.append(
                {
                    "time": int(i),
                    "action": current_policy["action"],
                    "quantity": round(
                        current_policy["quantity"], 2
                    ),
                    "participation": round(
                        current_policy["participation"], 4
                    ),
                    "urgency": round(
                        current_policy["urgency"], 3
                    ),
                    "confidence": round(
                        current_policy["confidence"], 3
                    ),
                    "rationale": current_policy["rationale"],
                }
            )

        policy = current_policy

        quantity = min(
            remaining,
            row["liquidity"] * policy["participation"],
            policy["quantity"],
        )

        # Spread the AI-selected block across the decision interval.
        quantity /= max(decision_interval, 1)

        if steps_left <= 2:
            quantity = min(
                remaining,
                max(
                    quantity,
                    remaining / max(steps_left, 1),
                ),
            )

        participation = quantity / max(row["liquidity"], 1)

        impact = (
            impact_coefficient
            * np.sqrt(max(participation, 0))
        )

        execution_price = row["ask"] * (1 + impact)
        cost = quantity * execution_price

        total_cost += cost
        remaining -= quantity

        records.append(
            {
                "time": i,
                "market_price": row["price"],
                "execution_price": execution_price,
                "quantity": quantity,
                "remaining": remaining,
                "liquidity": row["liquidity"],
                "volume": row["volume"],
                "spread": row["spread"],
                "volatility": row["volatility"],
                "participation": participation,
                "impact": impact,
                "action": policy["action"],
                "quality": calculate_market_quality(row),
                "confidence": policy["confidence"],
                "ai_urgency": policy["urgency"],
                "ai_rationale": policy["rationale"],
                "cost": cost,
            }
        )

    return pd.DataFrame(records), total_cost


# ============================================================
# MARKET QUALITY / EXECUTION ENGINE
# ============================================================

def calculate_market_quality(row, liquidity_scale=100_000, spread_scale=0.15,
                             volatility_scale=1.5):
    liquidity_score = min(row["liquidity"] / liquidity_scale, 1)
    spread_score = max(0, 1 - row["spread"] / spread_scale)
    volatility_score = max(0, 1 - row["volatility"] / volatility_scale)

    score = (
        0.45 * liquidity_score
        + 0.35 * spread_score
        + 0.20 * volatility_score
    )
    return float(np.clip(score, 0, 1))


def execution_decision(
    row,
    remaining,
    total_order,
    steps_left,
    max_participation=0.10,
    urgency=1.0,
):
    quality = calculate_market_quality(row)

    base_quantity = remaining / max(steps_left, 1)

    # Urgency > 1 makes the engine more willing to catch up.
    quality_multiplier = 0.70 + 0.70 * quality
    urgency_multiplier = 0.70 + 0.60 * urgency

    desired_quantity = base_quantity * quality_multiplier * urgency_multiplier

    liquidity_limit = row["liquidity"] * max_participation
    quantity = min(desired_quantity, liquidity_limit, remaining)

    # Deadline protection: progressively force completion as the
    # horizon gets short.
    if steps_left <= 8:
        quantity = min(
            remaining,
            max(quantity, remaining / max(steps_left, 1)),
        )

    if quality >= 0.70:
        action = "AGGRESSIVE"
    elif quality >= 0.45:
        action = "NORMAL"
    else:
        action = "CAUTIOUS"

    return {
        "quantity": float(max(quantity, 0)),
        "quality": quality,
        "action": action,
    }


def run_execution(
    market,
    total_order,
    strategy="Adaptive",
    max_participation=0.10,
    urgency=1.0,
    impact_coefficient=0.025,
):
    remaining = float(total_order)
    total_cost = 0.0
    records = []

    for i, row in market.iterrows():
        if remaining <= 0:
            break

        steps_left = len(market) - i

        if strategy == "Adaptive":
            decision = execution_decision(
                row,
                remaining,
                total_order,
                steps_left,
                max_participation,
                urgency,
            )
            quantity = decision["quantity"]
            action = decision["action"]
            quality = decision["quality"]
        elif strategy == "TWAP":
            quantity = remaining / steps_left
            action = "TWAP"
            quality = 0.5
        else:
            # A more aggressive benchmark that executes faster.
            quantity = min(
                remaining,
                (remaining / steps_left) * 1.35,
                row["liquidity"] * max_participation,
            )
            action = "AGGRESSIVE"
            quality = calculate_market_quality(row)

        participation = quantity / max(row["liquidity"], 1)

        # Square-root impact approximation.
        impact = impact_coefficient * np.sqrt(max(participation, 0))
        execution_price = row["ask"] * (1 + impact)

        cost = quantity * execution_price
        total_cost += cost
        remaining -= quantity

        records.append(
            {
                "time": i,
                "market_price": row["price"],
                "execution_price": execution_price,
                "quantity": quantity,
                "remaining": remaining,
                "liquidity": row["liquidity"],
                "volume": row["volume"],
                "spread": row["spread"],
                "volatility": row["volatility"],
                "participation": participation,
                "impact": impact,
                "action": action,
                "quality": quality,
                "cost": cost,
            }
        )

    result = pd.DataFrame(records)
    return result, total_cost


# ============================================================
# HELPERS
# ============================================================

def money(value):
    return f"₹{value:,.2f}"


def pct(value, digits=2):
    return f"{value * 100:.{digits}f}%"


def make_chart(fig, height=420):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(color="#cbd5e1"),
        xaxis=dict(gridcolor="#222a36", zeroline=False),
        yaxis=dict(gridcolor="#222a36", zeroline=False),
    )
    return fig


def build_decision_log(adaptive):
    rows = []

    if adaptive.empty:
        return pd.DataFrame()

    sample = adaptive.iloc[
        :: max(1, len(adaptive) // 12)
    ]

    for _, row in sample.iterrows():
        q = row["quality"]

        if q >= 0.70:
            reason = (
                "High market quality. Liquidity and spread conditions "
                "support increased execution intensity."
            )
        elif q >= 0.45:
            reason = (
                "Balanced conditions. Maintain the baseline execution "
                "pace while monitoring market quality."
            )
        else:
            reason = (
                "Deteriorating conditions. Reduce liquidity consumption "
                "and protect execution quality."
            )

        rows.append(
            {
                "Time": int(row["time"]),
                "Action": row["action"],
                "Quality": f"{q * 100:.1f}%",
                "Liquidity": f"{row['liquidity']:,.0f}",
                "Spread": f"{row['spread']:.4f}",
                "Volatility": f"{row['volatility']:.2f}",
                "Participation": f"{row['participation'] * 100:.2f}%",
                "Reason": reason,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SIDEBAR — CONTROL CENTER
# ============================================================

st.sidebar.markdown("## Execution Control Center")

if ai_client:
    st.sidebar.success("AI Analyst: Online")
else:
    st.sidebar.warning("AI Analyst: Offline")

st.sidebar.caption(
    "Sliders and numeric fields are synchronized. "
    "Type exact values when precision matters."
)


def synced_control(
    label,
    min_value,
    max_value,
    default,
    step,
    key_name,
    fmt=None,
    help_text=None,
):
    slider_key = f"{key_name}_slider"
    number_key = f"{key_name}_number"

    if slider_key not in st.session_state:
        st.session_state[slider_key] = default

    if number_key not in st.session_state:
        st.session_state[number_key] = default

    def slider_changed():
        st.session_state[number_key] = st.session_state[slider_key]

    def number_changed():
        st.session_state[slider_key] = st.session_state[number_key]

    c1, c2 = st.sidebar.columns([2.4, 1])

    with c1:
        st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=slider_key,
            format=fmt,
            help=help_text,
            on_change=slider_changed,
        )

    with c2:
        st.number_input(
            "Value",
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=number_key,
            format=fmt,
            label_visibility="collapsed",
            on_change=number_changed,
        )

    return st.session_state[number_key]


order_size = synced_control(
    "Institutional Order Size",
    10_000,
    10_000_000,
    1_000_000,
    10_000,
    "order_size",
    fmt="%.0f",
    help_text="Total synthetic order size.",
)

scenario = st.sidebar.selectbox(
    "Market Regime",
    list(SCENARIO_CONFIG.keys()),
)

seed = synced_control(
    "Market Seed",
    1,
    1000,
    42,
    1,
    "market_seed",
    fmt="%.0f",
    help_text="Use the same seed to reproduce a scenario exactly.",
)

steps = synced_control(
    "Execution Horizon",
    30,
    300,
    120,
    10,
    "execution_steps",
    fmt="%.0f",
    help_text="Number of synthetic market observations.",
)

max_participation = synced_control(
    "Maximum Participation",
    0.02,
    0.25,
    0.10,
    0.01,
    "max_participation",
    fmt="%.0f%%",
    help_text="Hard maximum share of available liquidity consumed per observation.",
)

urgency = synced_control(
    "Execution Urgency",
    0.50,
    2.00,
    1.00,
    0.05,
    "execution_urgency",
    fmt="%.2f",
    help_text="Fallback/default urgency and benchmark control.",
)

impact_coefficient = synced_control(
    "Impact Coefficient",
    0.005,
    0.060,
    0.025,
    0.005,
    "impact_coefficient",
    fmt="%.3f",
    help_text="Sensitivity of execution price to participation.",
)

st.sidebar.divider()

execution_mode = st.sidebar.radio(
    "Primary Execution Brain",
    ["AI Agent", "Mathematical Adaptive"],
    index=0,
    help=(
        "AI Agent lets the model choose the execution policy. "
        "Mathematical Adaptive uses the original deterministic rules."
    ),
)

ai_decision_interval = synced_control(
    "AI Decision Interval",
    1,
    20,
    5,
    1,
    "ai_decision_interval",
    fmt="%.0f",
    help_text=(
        "Observations between AI decisions. Lower values are more "
        "reactive but use more API calls."
    ),
)

run = st.sidebar.button(
    "Run Simulation",
    type="primary",
    use_container_width=True,
)

st.sidebar.divider()
st.sidebar.markdown("### Engine Inputs")
st.sidebar.markdown(
    """
    **Observed**
    - Price
    - Bid / ask spread
    - Volume
    - Liquidity
    - Volatility

    **AI Agent**
    - Current market context
    - Remaining order
    - Time remaining
    - Recent market history
    - Previous AI decisions

    **Hard Controls**
    - Maximum participation
    - Quantity boundary
    - Deadline protection
    """
)

# ============================================================
# RUN / STATE
# ============================================================

config_signature = (
    order_size,
    scenario,
    seed,
    steps,
    max_participation,
    urgency,
    impact_coefficient,
    execution_mode,
    ai_decision_interval,
)

if run or "config_signature" not in st.session_state or (
    st.session_state.config_signature != config_signature
):
    market = generate_market(seed, scenario, steps)

    if execution_mode == "AI Agent":
        adaptive, adaptive_cost = run_ai_agent(
            market,
            order_size,
            max_participation,
            urgency,
            impact_coefficient,
            int(ai_decision_interval),
        )
    else:
        adaptive, adaptive_cost = run_execution(
            market,
            order_size,
            "Adaptive",
            max_participation,
            urgency,
            impact_coefficient,
        )

    twap, twap_cost = run_execution(
        market,
        order_size,
        "TWAP",
        max_participation,
        urgency,
        impact_coefficient,
    )

    aggressive, aggressive_cost = run_execution(
        market,
        order_size,
        "Aggressive",
        max_participation,
        max(urgency, 1.25),
        impact_coefficient,
    )

    st.session_state.market = market
    st.session_state.adaptive = adaptive
    st.session_state.twap = twap
    st.session_state.aggressive = aggressive
    st.session_state.adaptive_cost = adaptive_cost
    st.session_state.twap_cost = twap_cost
    st.session_state.aggressive_cost = aggressive_cost
    st.session_state.config_signature = config_signature

market = st.session_state.market
adaptive = st.session_state.adaptive
twap = st.session_state.twap
aggressive = st.session_state.aggressive

adaptive_cost = st.session_state.adaptive_cost
twap_cost = st.session_state.twap_cost
aggressive_cost = st.session_state.aggressive_cost


# ============================================================
# DERIVED METRICS
# ============================================================

executed = adaptive["quantity"].sum()
remaining = max(order_size - executed, 0)

completion = executed / max(order_size, 1)
avg_execution = (
    (adaptive["quantity"] * adaptive["execution_price"]).sum()
    / max(executed, 1)
)
arrival_price = market.iloc[0]["price"]
slippage = avg_execution - arrival_price

twap_improvement = ((twap_cost - adaptive_cost) / max(twap_cost, 1)) * 100
aggressive_vs_adaptive = (
    (adaptive_cost - aggressive_cost) / max(adaptive_cost, 1)
) * 100

avg_participation = adaptive["participation"].mean()
avg_impact_bps = adaptive["impact"].mean() * 10_000
latest = market.iloc[-1]
latest_quality = calculate_market_quality(latest)

best_quality = market.apply(calculate_market_quality, axis=1).max()
worst_quality = market.apply(calculate_market_quality, axis=1).min()

latest_remaining = (
    adaptive.iloc[-1]["remaining"] if not adaptive.empty else order_size
)

latest_decision = execution_decision(
    latest,
    latest_remaining,
    order_size,
    max(len(market) - len(adaptive), 1),
    max_participation,
    urgency,
)

# ============================================================
# HERO
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        <div class="section-label">Execution Analytics / Simulation</div>
        <div class="hero-title">AI Adaptive Trade Execution Engine</div>
        <div class="hero-subtitle">
            A synthetic institutional execution simulator that dynamically
            balances urgency, liquidity consumption, spread, volatility,
            participation and market impact against benchmark strategies.
        </div>
        <br>
        <span class="pill">Regime: {scenario}</span>
        <span class="pill">Order: {order_size:,.0f} units</span>
        <span class="pill">Horizon: {steps} observations</span>
        <span class="pill">Seed: {seed}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# KPI ROW
# ============================================================

k1, k2, k3, k4, k5, k6 = st.columns(6)

k1.metric("Order Size", f"{order_size:,.0f}")
k2.metric("Arrival Price", money(arrival_price))
k3.metric("Avg Execution", money(avg_execution))
k4.metric("vs TWAP", f"{twap_improvement:+.2f}%")
k5.metric("Completion", pct(completion, 1))
k6.metric("Remaining", f"{remaining:,.0f}")

# ============================================================
# MARKET STATUS
# ============================================================

st.divider()
st.markdown('<div class="section-label">Market Snapshot</div>', unsafe_allow_html=True)

m1, m2, m3, m4, m5, m6 = st.columns(6)

m1.metric("Last Price", money(latest["price"]))
m2.metric("Bid", money(latest["bid"]))
m3.metric("Ask", money(latest["ask"]))
m4.metric("Spread", f"{latest['spread']:.4f}")
m5.metric("Liquidity", f"{latest['liquidity']:,.0f}")
m6.metric("Volatility", f"{latest['volatility']:.2f}")

# ============================================================
# DECISION + RISK
# ============================================================

st.divider()
decision_col, risk_col = st.columns([1.1, 1])

with decision_col:
    action = latest_decision["action"]

    if action == "AGGRESSIVE":
        action_class = "positive"
        reason = (
            "Current market quality is strong enough to increase execution "
            "intensity while staying inside the participation constraint."
        )
    elif action == "NORMAL":
        action_class = "neutral"
        reason = (
            "Market conditions are balanced. The engine is maintaining a "
            "moderate pace rather than chasing liquidity."
        )
    else:
        action_class = "negative"
        reason = (
            "Market quality has deteriorated. The engine is protecting "
            "execution quality by reducing liquidity consumption."
        )

    st.markdown(
        f"""
        <div class="decision-panel">
            <div class="section-label">Current Execution Decision</div>
            <div class="decision-action {action_class}">{action}</div>
            <div class="decision-reason">{reason}</div>
            <br>
            <span class="pill">Quality {latest_decision['quality'] * 100:.1f}%</span>
            <span class="pill">Suggested Qty {latest_decision['quantity']:,.0f}</span>
            <span class="pill">Participation Cap {max_participation * 100:.0f}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with risk_col:
    st.markdown(
        '<div class="section-label">Execution Risk Monitor</div>',
        unsafe_allow_html=True,
    )

    r1, r2 = st.columns(2)
    r1.metric("Avg Participation", f"{avg_participation * 100:.2f}%")
    r2.metric("Avg Impact", f"{avg_impact_bps:.2f} bps")
    r1.metric("Price Slippage", f"{slippage:+.4f}")
    r2.metric("Quality Range", f"{worst_quality * 100:.0f}%–{best_quality * 100:.0f}%")

    st.divider()
    st.subheader("AI Decision Explainer")

    if st.button("Explain Current Decision", key="ai_decision_btn"):
        with st.spinner("Analyzing decision..."):
            st.session_state["ai_decision"] = ai_decision_explanation(
                latest, latest_decision, latest_remaining, scenario
            )

    if "ai_decision" in st.session_state:
        st.info(st.session_state["ai_decision"])


# ============================================================
# TABS
# ============================================================

tab_overview, tab_execution, tab_microstructure, tab_compare, tab_log, tab_ai, tab_export = st.tabs(
    [
        "Overview",
        "Execution",
        "Market Microstructure",
        "Strategy Lab",
        "Decision Log",
        "Execution AI",
        "Data",
    ]
)

# ============================================================
# OVERVIEW TAB
# ============================================================

with tab_overview:
    left, right = st.columns(2)

    with left:
        st.subheader("Price and Execution")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=market["time"],
                y=market["price"],
                name="Market",
                mode="lines",
                line=dict(width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=adaptive["time"],
                y=adaptive["execution_price"],
                name="Adaptive",
                mode="markers+lines",
                marker=dict(size=5),
                line=dict(width=1),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=twap["time"],
                y=twap["execution_price"],
                name="TWAP",
                mode="markers",
                marker=dict(size=4),
            )
        )

        fig.update_layout(
            xaxis_title="Market Observation",
            yaxis_title="Price",
        )

        st.plotly_chart(make_chart(fig), use_container_width=True)

    with right:
        st.subheader("Cumulative Execution")

        fig2 = go.Figure()

        fig2.add_trace(
            go.Scatter(
                x=adaptive["time"],
                y=adaptive["quantity"].cumsum(),
                name="Adaptive",
                mode="lines",
                line=dict(width=3),
            )
        )

        fig2.add_trace(
            go.Scatter(
                x=twap["time"],
                y=twap["quantity"].cumsum(),
                name="TWAP",
                mode="lines",
            )
        )

        fig2.add_trace(
            go.Scatter(
                x=aggressive["time"],
                y=aggressive["quantity"].cumsum(),
                name="Aggressive",
                mode="lines",
                line=dict(dash="dash"),
            )
        )

        fig2.update_layout(
            xaxis_title="Market Observation",
            yaxis_title="Units Executed",
        )

        st.plotly_chart(make_chart(fig2), use_container_width=True)

    st.subheader("Execution Health")

    progress = min(completion, 1.0)
    st.progress(progress, text=f"Adaptive completion: {completion * 100:.1f}%")

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Adaptive Cost", money(adaptive_cost))
    h2.metric("TWAP Cost", money(twap_cost))
    h3.metric("Aggressive Cost", money(aggressive_cost))
    h4.metric("Adaptive Advantage", f"{twap_improvement:+.2f}%")

    st.divider()
    st.subheader("AI Execution Agent")

    if execution_mode == "AI Agent":
        st.success(
            f"AI controls execution policy every "
            f"{int(ai_decision_interval)} observations."
        )

        if "confidence" in adaptive.columns:
            latest_ai = adaptive.iloc[-1]
            a1, a2, a3, a4 = st.columns(4)

            a1.metric("Latest AI Action", str(latest_ai["action"]))
            a2.metric(
                "AI Confidence",
                f"{latest_ai['confidence'] * 100:.0f}%",
            )
            a3.metric(
                "AI Urgency",
                f"{latest_ai['ai_urgency']:.2f}x",
            )
            a4.metric(
                "AI Participation",
                f"{latest_ai['participation'] * 100:.2f}%",
            )

            st.info(latest_ai["ai_rationale"])
    else:
        st.info(
            "Mathematical Adaptive mode is active. "
            "Switch to AI Agent to let the model choose execution policy."
        )

    st.divider()
    st.subheader("Execution Intelligence")

    ctx = execution_context(
        market, adaptive, twap, aggressive,
        order_size, scenario, adaptive_cost, twap_cost, aggressive_cost
    )

    if st.button("Generate AI Execution Report", key="ai_report_btn"):
        with st.spinner("Analyzing execution..."):
            st.session_state["ai_report"] = ai_report(ctx)

    if "ai_report" in st.session_state:
        st.markdown(st.session_state["ai_report"])


# ============================================================
# EXECUTION TAB
# ============================================================

with tab_execution:
    st.subheader("Execution Schedule")

    schedule = adaptive.copy()
    schedule["Cumulative Executed"] = schedule["quantity"].cumsum()
    schedule["Execution %"] = schedule["Cumulative Executed"] / order_size * 100
    schedule["Impact (bps)"] = schedule["impact"] * 10_000

    st.dataframe(
        schedule[
            [
                "time",
                "action",
                "quantity",
                "Cumulative Executed",
                "Execution %",
                "execution_price",
                "participation",
                "Impact (bps)",
                "remaining",
            ]
        ].rename(
            columns={
                "time": "Time",
                "action": "Action",
                "quantity": "Quantity",
                "execution_price": "Execution Price",
                "participation": "Participation",
                "remaining": "Remaining",
            }
        ),
        use_container_width=True,
        hide_index=True,
        height=440,
    )

    st.subheader("Execution Pace")

    pace_fig = go.Figure()
    pace_fig.add_trace(
        go.Bar(
            x=adaptive["time"],
            y=adaptive["quantity"],
            name="Adaptive Quantity",
        )
    )
    pace_fig.update_layout(
        xaxis_title="Market Observation",
        yaxis_title="Units Executed",
    )
    st.plotly_chart(make_chart(pace_fig, 360), use_container_width=True)

# ============================================================
# MICROSTRUCTURE TAB
# ============================================================

with tab_microstructure:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Liquidity vs Participation")

        fig3 = go.Figure()
        fig3.add_trace(
            go.Scatter(
                x=market["time"],
                y=market["liquidity"],
                name="Available Liquidity",
                mode="lines",
            )
        )
        fig3.add_trace(
            go.Scatter(
                x=adaptive["time"],
                y=adaptive["quantity"],
                name="Adaptive Quantity",
                mode="lines",
            )
        )
        fig3.update_layout(
            xaxis_title="Market Observation",
            yaxis_title="Units",
        )
        st.plotly_chart(make_chart(fig3, 390), use_container_width=True)

    with c2:
        st.subheader("Volatility and Spread")

        fig4 = go.Figure()
        fig4.add_trace(
            go.Scatter(
                x=market["time"],
                y=market["volatility"],
                name="Volatility",
                mode="lines",
            )
        )
        fig4.add_trace(
            go.Scatter(
                x=market["time"],
                y=market["spread"],
                name="Spread",
                mode="lines",
            )
        )
        fig4.update_layout(
            xaxis_title="Market Observation",
            yaxis_title="Value",
        )
        st.plotly_chart(make_chart(fig4, 390), use_container_width=True)

    st.subheader("Market Quality")

    quality_series = market.apply(calculate_market_quality, axis=1)

    quality_fig = go.Figure()
    quality_fig.add_trace(
        go.Scatter(
            x=market["time"],
            y=quality_series * 100,
            name="Market Quality",
            mode="lines",
            fill="tozeroy",
        )
    )
    quality_fig.add_hline(y=70, line_dash="dash", annotation_text="Aggressive threshold")
    quality_fig.add_hline(y=45, line_dash="dash", annotation_text="Normal threshold")
    quality_fig.update_layout(
        xaxis_title="Market Observation",
        yaxis_title="Quality Score (%)",
        yaxis_range=[0, 100],
    )
    st.plotly_chart(make_chart(quality_fig, 380), use_container_width=True)

    st.subheader("Order Book Snapshot")

    book = pd.DataFrame(
        {
            "Level": ["Best Bid", "Mid Price", "Best Ask"],
            "Price": [
                latest["bid"],
                (latest["bid"] + latest["ask"]) / 2,
                latest["ask"],
            ],
            "Distance from Mid": [
                ((latest["bid"] - latest["price"]) / latest["price"]) * 100,
                0,
                ((latest["ask"] - latest["price"]) / latest["price"]) * 100,
            ],
        }
    )
    st.dataframe(
        book.style.format(
            {"Price": "₹{:.4f}", "Distance from Mid": "{:.4f}%"}
        ),
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# STRATEGY LAB
# ============================================================

with tab_compare:
    st.subheader("Strategy Benchmark")

    comparison = pd.DataFrame(
        {
            "Strategy": ["TWAP", "Adaptive", "Aggressive"],
            "Total Cost": [twap_cost, adaptive_cost, aggressive_cost],
            "Average Price": [
                twap["execution_price"].mean(),
                adaptive["execution_price"].mean(),
                aggressive["execution_price"].mean(),
            ],
            "Units Executed": [
                twap["quantity"].sum(),
                adaptive["quantity"].sum(),
                aggressive["quantity"].sum(),
            ],
            "Completion": [
                twap["quantity"].sum() / order_size,
                adaptive["quantity"].sum() / order_size,
                aggressive["quantity"].sum() / order_size,
            ],
            "Avg Impact (bps)": [
                twap["impact"].mean() * 10_000,
                adaptive["impact"].mean() * 10_000,
                aggressive["impact"].mean() * 10_000,
            ],
        }
    )

    st.dataframe(
        comparison.style.format(
            {
                "Total Cost": "₹{:,.2f}",
                "Average Price": "₹{:.4f}",
                "Units Executed": "{:,.0f}",
                "Completion": "{:.1%}",
                "Avg Impact (bps)": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Cost Comparison")

    cost_fig = go.Figure()
    cost_fig.add_trace(
        go.Bar(
            x=comparison["Strategy"],
            y=comparison["Total Cost"],
            text=[f"₹{x:,.0f}" for x in comparison["Total Cost"]],
            textposition="outside",
        )
    )
    cost_fig.update_layout(
        xaxis_title="Strategy",
        yaxis_title="Total Execution Cost",
        showlegend=False,
    )
    st.plotly_chart(make_chart(cost_fig, 360), use_container_width=True)

    st.subheader("What-If Controls")

    st.caption("Use exact numeric values for What-If experiments.")

    w1, w2, w3 = st.columns(3)

    with w1:
        what_if_urgency = st.number_input(
            "Test Urgency",
            min_value=0.50,
            max_value=2.00,
            value=float(urgency),
            step=0.05,
            format="%.2f",
        )

    with w2:
        what_if_participation = st.number_input(
            "Test Participation Cap",
            min_value=0.02,
            max_value=0.25,
            value=float(max_participation),
            step=0.01,
            format="%.2f",
        )

    with w3:
        what_if_impact = st.number_input(
            "Test Impact Coefficient",
            min_value=0.005,
            max_value=0.060,
            value=float(impact_coefficient),
            step=0.005,
            format="%.3f",
        )

    what_if, what_if_cost = run_execution(
        market,
        order_size,
        "Adaptive",
        what_if_participation,
        what_if_urgency,
        what_if_impact,
    )

    wi1, wi2, wi3, wi4 = st.columns(4)
    wi1.metric("What-If Cost", money(what_if_cost))
    wi2.metric(
        "Cost vs Current",
        f"{((adaptive_cost - what_if_cost) / max(adaptive_cost, 1)) * 100:+.2f}%",
    )
    wi3.metric(
        "What-If Completion",
        f"{what_if['quantity'].sum() / order_size * 100:.1f}%",
    )
    wi4.metric(
        "What-If Impact",
        f"{what_if['impact'].mean() * 10_000:.2f} bps",
    )

    st.caption(
        "What-if results are recalculated on the same synthetic market path, "
        "so you can isolate the effect of changing execution parameters."
    )

    if st.button("Ask AI to Evaluate This Scenario", key="ai_whatif_btn"):
        current = {
            "urgency": urgency,
            "participation_cap": max_participation,
            "impact_coefficient": impact_coefficient,
            "cost": adaptive_cost,
            "completion": adaptive["quantity"].sum() / order_size,
            "impact_bps": adaptive["impact"].mean() * 10000,
        }
        proposed = {
            "urgency": what_if_urgency,
            "participation_cap": what_if_participation,
            "impact_coefficient": what_if_impact,
            "cost": what_if_cost,
            "completion": what_if["quantity"].sum() / order_size,
            "impact_bps": what_if["impact"].mean() * 10000,
        }

        with st.spinner("Evaluating scenario..."):
            st.session_state["ai_whatif"] = ai_what_if(
                current, proposed
            )

    if "ai_whatif" in st.session_state:
        st.markdown(st.session_state["ai_whatif"])

# ============================================================
# DECISION LOG
# ============================================================

with tab_log:
    st.subheader("Adaptive Decision Log")

    log_df = build_decision_log(adaptive)
    st.dataframe(
        log_df,
        use_container_width=True,
        hide_index=True,
        height=470,
    )

    if execution_mode == "AI Agent" and "confidence" in adaptive.columns:
        st.subheader("AI Agent Decision History")

        ai_log = adaptive[
            [
                "time",
                "action",
                "quantity",
                "participation",
                "ai_urgency",
                "confidence",
                "ai_rationale",
            ]
        ].copy()

        ai_log = ai_log[
            ai_log["time"] % int(ai_decision_interval) == 0
        ]

        ai_log["participation"] *= 100
        ai_log["confidence"] *= 100

        st.dataframe(
            ai_log.rename(
                columns={
                    "time": "Time",
                    "action": "Action",
                    "quantity": "Quantity",
                    "participation": "Participation %",
                    "ai_urgency": "Urgency",
                    "confidence": "Confidence %",
                    "ai_rationale": "AI Rationale",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Decision Distribution")

    counts = adaptive["action"].value_counts()
    action_fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index,
                y=counts.values,
                text=counts.values,
                textposition="outside",
            )
        ]
    )
    action_fig.update_layout(
        xaxis_title="Execution Decision",
        yaxis_title="Observations",
        showlegend=False,
    )
    st.plotly_chart(make_chart(action_fig, 350), use_container_width=True)

    st.divider()
    st.subheader("AI Decision-Pattern Analysis")

    if st.button("Analyze Decision History", key="ai_history_btn"):
        with st.spinner("Analyzing decision history..."):
            st.session_state["ai_history"] = ai_history(adaptive)

    if "ai_history" in st.session_state:
        st.markdown(st.session_state["ai_history"])


# ============================================================
# EXECUTION AI TAB
# ============================================================

with tab_ai:
    st.subheader("Ask Execution AI")
    st.caption(
        "Ask questions about the current simulation. The AI receives "
        "the actual synthetic market and execution statistics."
    )

    ctx = execution_context(
        market, adaptive, twap, aggressive,
        order_size, scenario, adaptive_cost, twap_cost, aggressive_cost
    )

    if "ai_chat" not in st.session_state:
        st.session_state.ai_chat = []

    for message in st.session_state.ai_chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask about this execution...")

    if question:
        st.session_state.ai_chat.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                answer = call_execution_ai(
                    """
You are Execution AI inside an institutional execution simulator.

Answer questions about execution strategy, market quality,
liquidity, spread, volatility, market impact, participation,
TWAP, Adaptive, Aggressive execution, completion, cost and
decision logic.

Use ONLY supplied simulation data. Never invent values.
If the data cannot answer something, say so.

This is synthetic data, not live market data or financial advice.
""",
                    f"SIMULATION:\n{ctx}\n\nQUESTION:\n{question}",
                )
                st.markdown(answer)

        st.session_state.ai_chat.append(
            {"role": "assistant", "content": answer}
        )

    st.divider()
    st.markdown("### Quick Analysis")

    q1, q2, q3 = st.columns(3)

    if q1.button("Summarize Simulation", use_container_width=True):
        with st.spinner("Generating summary..."):
            st.session_state["ai_quick"] = ai_report(ctx)

    if q2.button("Compare Adaptive vs TWAP", use_container_width=True):
        with st.spinner("Comparing strategies..."):
            st.session_state["ai_quick"] = call_execution_ai(
                """
Compare Adaptive and TWAP using only the supplied simulation data.
Explain the main reason for any cost or behavior difference.
Keep the answer under 200 words.
""",
                f"SIMULATION:\n{ctx}",
            )

    if q3.button("Identify Main Risk", use_container_width=True):
        with st.spinner("Identifying risk..."):
            st.session_state["ai_quick"] = call_execution_ai(
                """
Identify the single most important execution risk in this
synthetic simulation and explain the evidence. Do not invent data.
Keep the answer under 150 words.
""",
                f"SIMULATION:\n{ctx}",
            )

    if "ai_quick" in st.session_state:
        st.markdown(st.session_state["ai_quick"])


# ============================================================
# DATA / EXPORT
# ============================================================

with tab_export:
    st.subheader("Simulation Data")

    data_choice = st.radio(
        "Dataset",
        ["Adaptive", "TWAP", "Market"],
        horizontal=True,
    )

    selected_data = {
        "Adaptive": adaptive,
        "TWAP": twap,
        "Market": market,
    }[data_choice]

    st.dataframe(
        selected_data,
        use_container_width=True,
        hide_index=True,
        height=430,
    )

    csv = selected_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV",
        data=csv,
        file_name=f"{data_choice.lower()}_execution_data.csv",
        mime="text/csv",
        use_container_width=False,
    )

    st.subheader("Configuration")

    config_df = pd.DataFrame(
        {
            "Parameter": [
                "Order Size",
                "Market Regime",
                "Seed",
                "Execution Horizon",
                "Maximum Participation",
                "Execution Urgency",
                "Impact Coefficient",
            ],
            "Value": [
                f"{order_size:,.0f}",
                scenario,
                seed,
                steps,
                f"{max_participation * 100:.1f}%",
                f"{urgency:.2f}x",
                f"{impact_coefficient:.3f}",
            ],
        }
    )

    st.dataframe(
        config_df,
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# ARCHITECTURE
# ============================================================

st.divider()
st.markdown('<div class="section-label">System Architecture</div>', unsafe_allow_html=True)
st.subheader("How the engine works")

a1, a2, a3, a4, a5 = st.columns(5)

with a1:
    st.markdown(
        """
        **01 — Observe**

        Monitor price, spread, volume, liquidity and volatility on every
        synthetic market observation.
        """
    )

with a2:
    st.markdown(
        """
        **02 — Understand**

        The AI evaluates liquidity, spread, volatility, order state,
        time remaining and recent execution outcomes.
        """
    )

with a3:
    st.markdown(
        """
        **03 — Decide**

        The AI chooses action, quantity, participation, urgency and
        confidence for the next execution block.
        """
    )

with a4:
    st.markdown(
        """
        **04 — Guard**

        A deterministic safety layer enforces hard participation,
        quantity and deadline constraints.
        """
    )

with a5:
    st.markdown(
        """
        **05 — Evaluate**

        Measure cost, impact and completion against TWAP, Aggressive
        and the original mathematical strategy.
        """
    )

st.markdown(
    """
    <div class="footer">
        Prototype environment • Synthetic market data • AI-assisted execution • No live orders are sent
        <br>
        This simulator is for experimentation and execution-model research.
    </div>
    """,
    unsafe_allow_html=True,
)
