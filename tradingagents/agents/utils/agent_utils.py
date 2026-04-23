from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Only applied to user-facing agents (analysts, portfolio manager).
    Internal debate agents stay in English for reasoning quality.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def build_instrument_context(ticker: str) -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    return (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`)."
    )


def build_portfolio_context(portfolio_str: str) -> str:
    """Format user portfolio with auto-calculated weights and verification.

    All derived numbers (position value, weight, P&L, totals) are computed here
    so the LLM agent receives pre-verified arithmetic and never has to calculate.

    Returns empty string when no portfolio is provided (backward compatible).
    """
    if not portfolio_str or not portfolio_str.strip():
        return ""

    import json
    try:
        portfolio = json.loads(portfolio_str)
    except (json.JSONDecodeError, TypeError):
        return f"\n\n**User Portfolio (raw):** {portfolio_str}"

    lines = ["\n\n**User's Current Portfolio (all values in USD):**"]

    # --- Compute each position ---
    computed_positions = []
    for pos in portfolio.get("positions", []):
        ticker = pos.get("ticker", "Unknown")
        shares = pos.get("shares", 0)
        cost_per = pos.get("cost_basis_usd", 0)  # per-share cost
        price_per = pos.get("current_price_usd", 0)  # per-share current price
        val_override = pos.get("current_value_usd", 0)  # total value override

        # Compute total value
        if val_override and not price_per:
            current_value = val_override
        elif price_per and shares:
            current_value = price_per * shares
        elif val_override:
            current_value = val_override
        else:
            current_value = 0

        # Compute total cost
        cost_total = cost_per * shares if (cost_per and shares) else 0

        # P&L
        if cost_total > 0 and current_value > 0:
            pnl_usd = current_value - cost_total
            pnl_pct = pnl_usd / cost_total * 100
        else:
            pnl_usd = 0
            pnl_pct = 0

        computed_positions.append({
            "ticker": ticker,
            "shares": shares,
            "cost_per": cost_per,
            "price_per": price_per,
            "cost_total": cost_total,
            "current_value": current_value,
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct,
        })

    # --- Compute totals ---
    total_position_value = sum(p["current_value"] for p in computed_positions)
    total_cost = sum(p["cost_total"] for p in computed_positions)
    cash = portfolio.get("cash_usd", 0)
    total_portfolio = total_position_value + cash
    total_pnl = total_position_value - total_cost

    # Override total if user provided it and it's close enough
    user_total = portfolio.get("total_portfolio_usd", 0)
    if user_total and not total_portfolio:
        total_portfolio = user_total

    # --- Format each position with weight ---
    for p in computed_positions:
        weight = p["current_value"] / total_portfolio * 100 if total_portfolio else 0
        parts = [f"{p['ticker']}:"]
        if p["shares"]:
            parts.append(f"{p['shares']} shares")
        if p["price_per"] and p["shares"]:
            parts.append(f"@ ${p['price_per']:.2f}")
        if p["current_value"]:
            parts.append(f"= ${p['current_value']:,.0f}")
        if p["cost_total"]:
            parts.append(f"(cost ${p['cost_total']:,.0f})")
        if p["pnl_pct"]:
            parts.append(f"P&L ${p['pnl_usd']:+,.0f} ({p['pnl_pct']:+.1f}%)")
        parts.append(f"[{weight:.1f}% of portfolio]")
        lines.append(f"  - {' | '.join(parts)}")

    # --- Cash & totals ---
    if cash:
        cash_weight = cash / total_portfolio * 100 if total_portfolio else 0
        lines.append(f"  - Cash: ${cash:,.0f} [{cash_weight:.1f}%]")

    lines.append(f"  ---")
    lines.append(f"  - Total portfolio value: ${total_portfolio:,.0f}")
    lines.append(f"  - Total invested: ${total_cost:,.0f} | Unrealized P&L: ${total_pnl:+,.0f}")

    # --- Verification ---
    lines.append(f"  - Verify: positions ${total_position_value:,.0f} + cash ${cash:,.0f} = ${total_portfolio:,.0f} ✓")

    risk = portfolio.get("risk_tolerance")
    if risk:
        lines.append(f"  - Risk tolerance: {risk}")

    # --- Concentration warning ---
    top = max(computed_positions, key=lambda p: p["current_value"]) if computed_positions else None
    if top and total_portfolio:
        top_weight = top["current_value"] / total_portfolio * 100
        if top_weight > 30:
            lines.append(
                f"\n  ⚠ Concentration: {top['ticker']} represents {top_weight:.1f}% of portfolio. "
                "Adding to this position significantly increases risk."
            )

    lines.append(
        "\nFactor this portfolio into your decision: consider concentration risk, "
        "correlation with existing positions, and whether the recommendation "
        "complements or overloads the current allocation."
    )

    return "\n".join(lines)

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
