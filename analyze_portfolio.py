"""Portfolio-level multi-agent debate analysis.

Usage: python analyze_portfolio.py [--portfolio my_portfolio.json] [--date 2026-04-23]

Architecture (mirrors TradingAgents multi-agent debate):
  Phase 1: Concentration Efficiency vs Resilience analysis → Portfolio Architect judges
  Phase 2: Position Evaluator → Opportunity Cost/Downside/Risk-Adjusted analysis → CIO judges

Each agent has an analytical lens (not a pre-determined conclusion).
"""

import os, sys, json, time, argparse, glob
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()
os.environ['http_proxy'] = 'http://localhost:7890'
os.environ['https_proxy'] = 'http://localhost:7890'

import yfinance as yf

PORTFOLIO_FILE = "my_portfolio.json"
REPORT_DIR = "results/portfolio"


# ═══════════════════════════════════════════
# Data Collection
# ═══════════════════════════════════════════

def load_portfolio(path):
    with open(path) as f:
        return json.load(f)


def get_existing_report(ticker):
    """Find the most recent single-stock analysis report."""
    files = sorted(glob.glob(f"results/{ticker}/latest/signal_*.json"), reverse=True)
    if not files:
        return None
    with open(files[0]) as f:
        data = json.load(f)
    return {
        'date': data.get('date', ''),
        'signal': data.get('signal', ''),
        'decision': data.get('final_trade_decision', ''),
        'plan': data.get('investment_plan', ''),
    }


def fetch_stock_data(ticker):
    """Fetch basic metrics via yfinance."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        hist = t.history(period="1mo")
        price = info.get('currentPrice') or info.get('regularMarketPrice')
        if price is None and not hist.empty:
            price = hist['Close'].iloc[-1]
        perf_1m = None
        if not hist.empty and len(hist) >= 2:
            perf_1m = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
        return {
            'ticker': ticker,
            'current_price': round(price, 2) if price else None,
            'trailing_pe': round(info['trailingPE'], 1) if info.get('trailingPE') else None,
            'forward_pe': round(info['forwardPE'], 1) if info.get('forwardPE') else None,
            'peg': round(info['pegRatio'], 2) if info.get('pegRatio') else None,
            '52w_high': round(info['fiftyTwoWeekHigh'], 2) if info.get('fiftyTwoWeekHigh') else None,
            '52w_low': round(info['fiftyTwoWeekLow'], 2) if info.get('fiftyTwoWeekLow') else None,
            '1m_return_pct': round(perf_1m, 1) if perf_1m else None,
        }
    except Exception as e:
        return {'ticker': ticker, 'error': str(e)[:200]}


def classify_weight(pct):
    if pct >= 10:
        return "MAJOR", "detailed analysis warranted"
    elif pct >= 5:
        return "STANDARD", "medium analysis"
    else:
        return "MINOR", "brief treatment"


def build_position_cards(portfolio, stock_data, reports):
    """Build weight-tiered position cards for all agents."""
    positions = portfolio.get('positions', [])
    cash = portfolio.get('cash_usd', 0)

    # Compute total portfolio value
    total = cash
    for pos in positions:
        ticker = pos['ticker']
        sd = next((s for s in stock_data if s['ticker'] == ticker), {})
        price = sd.get('current_price') or pos.get('current_price_usd', 0)
        total += pos.get('shares', 0) * price

    cards = []
    for pos in positions:
        ticker = pos['ticker']
        shares = pos.get('shares', 0)
        cost = pos.get('cost_basis_usd', 0)
        sd = next((s for s in stock_data if s['ticker'] == ticker), {})
        price = sd.get('current_price') or pos.get('current_price_usd', 0)
        value = shares * price
        cost_total = shares * cost
        weight = value / total * 100 if total else 0
        pnl = value - cost_total
        pnl_pct = (pnl / cost_total * 100) if cost_total else 0

        tier, treatment = classify_weight(weight)
        card = f"### {ticker} [{weight:.1f}%] -- {tier} ({treatment})\n"
        card += f"- {shares} shares @ ${price:.2f} = ${value:,.0f} (cost ${cost_total:,.0f}, P&L ${pnl:+,.0f} / {pnl_pct:+.1f}%)\n"

        # Add metrics from yfinance
        metrics = []
        if sd.get('trailing_pe'): metrics.append(f"PE {sd['trailing_pe']}")
        if sd.get('forward_pe'): metrics.append(f"Fwd PE {sd['forward_pe']}")
        if sd.get('peg'): metrics.append(f"PEG {sd['peg']}")
        if sd.get('52w_high'): metrics.append(f"52w ${sd['52w_low']:.0f}-${sd['52w_high']:.0f}")
        if sd.get('1m_return_pct') is not None: metrics.append(f"1mo {sd['1m_return_pct']:+.1f}%")
        if sd.get('error'): metrics.append(f"[data error]")
        if metrics:
            card += f"- Metrics: {' | '.join(metrics)}\n"

        # Add cached agent report based on weight tier
        report = reports.get(ticker)
        if report:
            if tier == "MAJOR":
                excerpt = report['decision'][:1500]
                card += f"- Agent Report ({report['date']}, Signal: {report['signal']}):\n{excerpt}\n"
            elif tier == "STANDARD":
                excerpt = report['decision'][:800]
                card += f"- Agent Signal: {report['signal']} ({report['date']}): {excerpt}\n"
            else:
                card += f"- Agent Signal: {report['signal']} ({report['date']})\n"

        cards.append(card)

    # Cash
    cash_weight = cash / total * 100 if total else 0
    cash_tier, _ = classify_weight(cash_weight)
    cards.append(f"### CASH [{cash_weight:.1f}%] -- {cash_tier}\n")
    cards.append(f"- ${cash:,.0f}")

    return "\n".join(cards), total


# ═══════════════════════════════════════════
# LLM Setup
# ═══════════════════════════════════════════

def create_llm_clients(config):
    from tradingagents.llm_clients.factory import create_llm_client
    deep = create_llm_client(
        provider=config["llm_provider"], model=config["deep_think_llm"],
        base_url=config.get("backend_url"),
    )
    quick = create_llm_client(
        provider=config["llm_provider"], model=config["quick_think_llm"],
        base_url=config.get("backend_url"),
    )
    return deep.get_llm(), quick.get_llm()


def invoke(llm, prompt, label, state):
    """Invoke LLM with timing."""
    t0 = time.time()
    response = llm.invoke(prompt)
    elapsed = time.time() - t0
    state["step_timings"][label] = round(elapsed, 1)
    print(f"  [{label}] {elapsed:.0f}s", flush=True)
    return response.content


# ═══════════════════════════════════════════
# Phase 1: Portfolio Structure Debate
# ═══════════════════════════════════════════

def run_concentration_analyst(state, llm):
    user_ctx = _user_context(state)
    prompt = f"""You are a Concentration Efficiency Analyst. Your analytical lens: evaluate whether each position is earning its allocation — is the portfolio's concentration delivering returns commensurate with the risk of being concentrated?

{user_ctx}

PORTFOLIO POSITIONS:
{state['position_cards']}

{state['portfolio_context']}

Analyze from this perspective:
- For each major holding, does its performance (P&L, momentum, valuation) justify its current weight?
- Is the thematic overlap (e.g., AI/semiconductors) a coherent investment thesis or accidental correlation?
- Is the cash allocation consistent with the stated cash_strategy, or does it indicate indecision?
- What would be the cost (in expected return) of diversifying away from current winners?

Use actual weights, P&L figures, and metrics from the data. Let the numbers drive your conclusions — you may find concentration is efficient, inefficient, or mixed. 3-5 paragraphs."""

    content = invoke(llm, prompt, "concentration", state)
    state["concentration_arg"] = f"Concentration Analyst: {content}"
    state["phase1_history"] += state["concentration_arg"]


def run_diversification_analyst(state, llm):
    user_ctx = _user_context(state)
    prompt = f"""You are a Portfolio Resilience Analyst. Your analytical lens: stress-test this portfolio's structure — how would it perform under various market regimes (sector rotation, rate shock, recession, theme fatigue)?

{user_ctx}

PORTFOLIO POSITIONS:
{state['position_cards']}

{state['portfolio_context']}

PREVIOUS ANALYSIS:
{state['phase1_history']}

Analyze from this perspective:
- Correlation structure: which holdings are likely to move together, and under what scenarios?
- Sector/geographic coverage: are there meaningful risk exposures the portfolio is missing?
- Single-stock tail risk: what happens to portfolio value if the largest holding drops 20-30%?
- If you find the current structure is resilient, say so. If you find vulnerabilities, quantify them.
- Where appropriate, suggest weight ranges that would improve risk-adjusted outcomes — but only if the data supports it.

Engage with the previous analyst's points where relevant. Use specific weights and dollar amounts. 3-5 paragraphs."""

    content = invoke(llm, prompt, "diversification", state)
    state["diversification_arg"] = f"Resilience Analyst: {content}"
    state["phase1_history"] += "\n\n" + state["diversification_arg"]


def run_portfolio_architect(state, llm):
    user_ctx = _user_context(state)
    prompt = f"""You are a Portfolio Architect. Your role: synthesize the two structural analyses below into a definitive assessment. Evaluate both arguments on their merits — data quality, logical coherence, and relevance to this investor's stated goals.

{user_ctx}

PORTFOLIO:
{state['position_cards']}

{state['portfolio_context']}

ANALYSES:
{state['phase1_history']}

REQUIRED OUTPUT:

## Structural Verdict
[Sound / Needs minor adjustment / Needs major rebalancing — with reasoning]

## Weight Targets
For each position: CURRENT -> RECOMMENDED RANGE (one-line rationale each)
- NVDA: 22.6% -> X%-Y% (reason)
[repeat for each]

## Key Structural Issues
[Top 3 concerns, numbered]

## Cash Assessment
[Is the cash level appropriate given the cash_strategy?]

## Analysis Notes
[Which analyst made stronger points and why. Acknowledge where the data was ambiguous.]

Be decisive. Do not default to 'it depends'."""

    state["structural_verdict"] = invoke(llm, prompt, "architect", state)


# ═══════════════════════════════════════════
# Phase 2: Position-Level Analysis & Risk Debate
# ═══════════════════════════════════════════

def run_position_evaluator(state, llm):
    user_ctx = _user_context(state)
    prompt = f"""You are a Position Evaluator. Given the Portfolio Architect's structural verdict, produce per-position action recommendations.

STRUCTURAL VERDICT:
{state['structural_verdict']}

PORTFOLIO:
{state['position_cards']}

{state['portfolio_context']}

{user_ctx}

INSTRUCTIONS -- Analysis depth MUST match position weight:

MAJOR (>10%): Full analysis
- Verdict: STRONG BUY / BUY / HOLD / REDUCE / SELL
- Rationale: 3-4 sentences referencing data, P&L, valuation
- Key catalyst or risk in next 30 days
- If agent report exists, reconcile your verdict with the agent's signal

STANDARD (5-10%): Medium analysis
- Verdict: BUY / HOLD / REDUCE / SELL
- Rationale: 2 sentences
- One key risk

MINOR (<5%): Brief treatment
- Verdict: HOLD / ADD / EXIT (one line only)

CASH: Assess based on cash_strategy.

Format:
### [TICKER] (X.X%) -- Verdict: HOLD
[Analysis at appropriate depth]"""

    state["position_verdicts"] = invoke(llm, prompt, "evaluator", state)


def run_aggressive_analyst(state, llm):
    prompt = f"""You are an Opportunity Cost Analyst. Your analytical lens: evaluate whether this portfolio is capturing available upside efficiently, or whether conservative positioning is leaving returns on the table.

PORTFOLIO: {state['position_cards']}

POSITION VERDICTS: {state['position_verdicts']}

STRUCTURAL VERDICT: {state['structural_verdict']}

Analyze from this perspective:
- For positions with strong momentum / low valuations / clear catalysts: is the current sizing capturing the opportunity or underweighting it?
- For positions with weak fundamentals or negative momentum: is holding them justified, or would reallocating capital improve expected returns?
- For the cash allocation: given current market conditions, is it earning its keep or creating a drag?
- Consider the investor's stated risk tolerance — an aggressive recommendation for a conservative investor may be inappropriate.

Use specific position sizes and dollar amounts. Your conclusions should follow from the data, not from a presumption that action is needed. 3-4 paragraphs."""

    content = invoke(llm, prompt, "aggressive", state)
    state["aggressive_risk_arg"] = f"Opportunity Cost Analyst: {content}"
    state["phase2_history"] += state["aggressive_risk_arg"]


def run_defensive_analyst(state, llm):
    prompt = f"""You are a Downside Risk Analyst. Your analytical lens: identify and quantify the risks in this portfolio — what could go wrong, how badly, and how likely is it?

PORTFOLIO: {state['position_cards']}

POSITION VERDICTS: {state['position_verdicts']}

STRUCTURAL VERDICT: {state['structural_verdict']}

RISK DEBATE SO FAR:
{state['phase2_history']}

Analyze from this perspective:
- Tail risk scenarios: what events could cause a 20%+ portfolio drawdown? How concentrated is the exposure?
- For positions with extended valuations, high beta, or crowded trades: what's the realistic downside?
- For the cash buffer: what risks does it mitigate vs. what opportunities does it forgo?
- If you find the portfolio is already well-hedged, say so. If you find vulnerabilities, quantify the potential loss.

Where the previous analyst's recommendations increase risk, evaluate whether the incremental upside justifies the incremental downside. Use specific dollar amounts. 3-4 paragraphs."""

    content = invoke(llm, prompt, "defensive", state)
    state["defensive_risk_arg"] = f"Downside Risk Analyst: {content}"
    state["phase2_history"] += "\n\n" + state["defensive_risk_arg"]


def run_pragmatic_analyst(state, llm):
    prompt = f"""You are a Risk-Adjusted Return Analyst. Your analytical lens: evaluate this portfolio on a risk-adjusted basis — are you getting the best return per unit of risk?

PORTFOLIO: {state['position_cards']}

POSITION VERDICTS: {state['position_verdicts']}

STRUCTURAL VERDICT: {state['structural_verdict']}

RISK DEBATE SO FAR:
{state['phase2_history']}

Analyze from this perspective:
- Where do the two previous analysts agree? Those points are likely robust.
- Where do they disagree? Evaluate which side the data supports.
- For each major position, what is the risk-adjusted case for its current sizing?
- Propose specific position adjustments that optimize the risk/return tradeoff — this may mean reducing risk, increasing risk, or maintaining the status quo depending on what the data shows.

Let the evidence determine your recommendation, not a default to "split the difference." 3-4 paragraphs."""

    content = invoke(llm, prompt, "pragmatic", state)
    state["pragmatic_risk_arg"] = f"Risk-Adjusted Analyst: {content}"
    state["phase2_history"] += "\n\n" + state["pragmatic_risk_arg"]


def run_cio_final(state, llm):
    user_ctx = _user_context(state)
    prompt = f"""You are the Chief Investment Officer. Your role: synthesize all analyses into a final portfolio recommendation. Evaluate each analyst's contribution on data quality, logical coherence, and relevance to this investor's stated goals and risk tolerance.

{user_ctx}

PORTFOLIO OVERVIEW:
{state['portfolio_context']}

POSITIONS:
{state['position_cards']}

PHASE 1 - STRUCTURAL ANALYSES:
{state['phase1_history']}

PORTFOLIO ARCHITECT'S VERDICT:
{state['structural_verdict']}

PHASE 2 - RISK ANALYSES:
{state['phase2_history']}

POSITION VERDICTS:
{state['position_verdicts']}

Produce EXACTLY these sections:

## 1. Portfolio Score: X/10
[One paragraph]

## 2. Concentration Analysis
[2-3 paragraphs -- evaluate concentration in context of the investment_style and debate evidence]

## 3. Per-Position Verdicts
[For each position, analysis depth matching weight tier. MINOR positions get ONE line only.]

## 4. Rebalancing Plan
[Specific actions: what to buy/sell/hold, amounts, priority order. Only recommend changes the data supports.]

## 5. Risk Scenarios
[Bull case / Base case / Bear case with estimated portfolio impact]

## 6. Priority Actions This Week
[Top 5 numbered items with specific tickers and dollar amounts]

Be decisive. Ground every recommendation in debate evidence and actual portfolio data. Note where analysts agreed (high confidence) vs. disagreed (lower confidence)."""

    state["final_report"] = invoke(llm, prompt, "CIO", state)


# ═══════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════

def _user_context(state):
    p = state.get("portfolio_raw", {})
    parts = []
    if p.get("risk_tolerance"):
        parts.append(f"Risk tolerance: {p['risk_tolerance']}")
    if p.get("investment_style"):
        parts.append(f"Investment style: {p['investment_style']}")
    if p.get("cash_strategy"):
        parts.append(f"Cash strategy: {p['cash_strategy']}")
    return "USER CONTEXT:\n" + "\n".join(f"- {x}" for x in parts) if parts else ""


# ═══════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--portfolio', default=PORTFOLIO_FILE)
    parser.add_argument('--date', default=datetime.now().strftime('%Y-%m-%d'))
    args = parser.parse_args()

    # --- Step 0: Data Collection ---
    print(f"Loading portfolio from {args.portfolio}...", flush=True)
    portfolio = load_portfolio(args.portfolio)
    tickers = [p['ticker'] for p in portfolio.get('positions', [])]

    print(f"\nChecking for cached reports...", flush=True)
    reports = {}
    for ticker in tickers:
        report = get_existing_report(ticker)
        if report:
            print(f"  {ticker}: {report['signal']} ({report['date']})", flush=True)
            reports[ticker] = report
        else:
            print(f"  {ticker}: no report", flush=True)

    print(f"\nFetching market data...", flush=True)
    stock_data = []
    for ticker in tickers:
        if ticker in reports:
            print(f"  {ticker}: using cached", flush=True)
            stock_data.append({'ticker': ticker})
        else:
            print(f"  {ticker}: fetching...", end="", flush=True)
            data = fetch_stock_data(ticker)
            print(f" ${data.get('current_price', 'ERR')}", flush=True)
            stock_data.append(data)

    # Build position cards
    position_cards, total_value = build_position_cards(portfolio, stock_data, reports)

    # Build portfolio context
    from tradingagents.agents.utils.agent_utils import build_portfolio_context
    portfolio_context = build_portfolio_context(json.dumps(portfolio))

    # Create LLM clients
    from tradingagents.default_config import DEFAULT_CONFIG
    config = DEFAULT_CONFIG.copy()
    deep_llm, quick_llm = create_llm_clients(config)

    # Initialize state
    state = {
        "portfolio_raw": portfolio,
        "position_cards": position_cards,
        "portfolio_context": portfolio_context,
        "stock_data": stock_data,
        "reports": reports,
        "phase1_history": "",
        "concentration_arg": "",
        "diversification_arg": "",
        "structural_verdict": "",
        "phase2_history": "",
        "position_verdicts": "",
        "aggressive_risk_arg": "",
        "defensive_risk_arg": "",
        "pragmatic_risk_arg": "",
        "final_report": "",
        "step_timings": {},
    }

    # --- Run Pipeline ---
    total_start = time.time()

    print(f"\n{'='*50}", flush=True)
    print(f"PHASE 1: Portfolio Structure Debate", flush=True)
    print(f"{'='*50}", flush=True)
    run_concentration_analyst(state, quick_llm)
    run_diversification_analyst(state, quick_llm)
    run_portfolio_architect(state, deep_llm)

    print(f"\n{'='*50}", flush=True)
    print(f"PHASE 2: Position Analysis & Risk Debate", flush=True)
    print(f"{'='*50}", flush=True)
    run_position_evaluator(state, quick_llm)
    run_aggressive_analyst(state, quick_llm)
    run_defensive_analyst(state, quick_llm)
    run_pragmatic_analyst(state, quick_llm)
    run_cio_final(state, deep_llm)

    total_elapsed = time.time() - total_start

    # --- Save ---
    os.makedirs(REPORT_DIR, exist_ok=True)

    report = {
        'date': args.date,
        'total_elapsed': round(total_elapsed, 1),
        'step_timings': state["step_timings"],
        'portfolio': portfolio,
        'final_report': state["final_report"],
        'structural_verdict': state["structural_verdict"],
        'position_verdicts': state["position_verdicts"],
    }
    with open(f"{REPORT_DIR}/report_{args.date}.json", 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    debug = {
        'date': args.date,
        'phase1_history': state["phase1_history"],
        'phase2_history': state["phase2_history"],
        'concentration_arg': state["concentration_arg"],
        'diversification_arg': state["diversification_arg"],
        'aggressive_risk_arg': state["aggressive_risk_arg"],
        'defensive_risk_arg': state["defensive_risk_arg"],
        'pragmatic_risk_arg': state["pragmatic_risk_arg"],
        'position_cards': state["position_cards"],
        'portfolio_context': state["portfolio_context"],
        'step_timings': state["step_timings"],
    }
    with open(f"{REPORT_DIR}/debate_{args.date}.json", 'w') as f:
        json.dump(debug, f, indent=2, ensure_ascii=False)

    # --- Print Results ---
    print(f"\n{'='*60}")
    print(f"PORTFOLIO ANALYSIS | {total_elapsed:.0f}s")
    print(f"{'='*60}")
    for step, t in state["step_timings"].items():
        print(f"  {step}: {t}s")
    print(f"\n{state['final_report']}")
    print(f"\nSaved: {REPORT_DIR}/report_{args.date}.json")
    print(f"Debug: {REPORT_DIR}/debate_{args.date}.json")


if __name__ == '__main__':
    main()
