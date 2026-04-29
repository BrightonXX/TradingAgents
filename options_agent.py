"""Options Analysis Agent — analyze options chains and recommend strategies.

Usage:
    python options_agent.py                     # all holdings
    python options_agent.py NVDA                # single ticker
    python options_agent.py NVDA MSFT MU        # multiple tickers
    python options_agent.py --sell-put NVDA     # sell put recommendations
    python options_agent.py --covered-call NVDA # covered call recommendations
    python options_agent.py --all NVDA          # full analysis
"""
import os, sys, json, argparse
from datetime import datetime, timedelta

os.environ.setdefault('http_proxy', 'http://localhost:7890')
os.environ.setdefault('https_proxy', 'http://localhost:7890')

import yfinance as yf

# ── Config ──────────────────────────────────────────────────────────────
PORTFOLIO_FILE = "my_portfolio.json"

# Colors
class C:
    RST = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GRN = "\033[32m"
    YEL = "\033[33m"
    BLU = "\033[34m"
    CYN = "\033[36m"
    BRED = "\033[1;31m"
    BGRN = "\033[1;32m"
    BYEL = "\033[1;33m"
    BBLU = "\033[1;34m"
    MAG = "\033[35m"
    BMAG = "\033[1;35m"


# ── Data ────────────────────────────────────────────────────────────────

def load_portfolio():
    try:
        return json.load(open(PORTFOLIO_FILE))
    except:
        return None


def get_options_chain(ticker, expiration=None):
    """Fetch options chain for a ticker."""
    t = yf.Ticker(ticker)
    exps = t.options
    if not exps:
        return None, None, None

    # Pick expiration: closest to 30 DTE if not specified
    if expiration is None:
        target = datetime.now() + timedelta(days=30)
        exp = min(exps, key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d") - target).days))
    else:
        exp = expiration

    chain = t.option_chain(exp)
    info = t.info or {}
    spot = info.get("currentPrice") or info.get("regularMarketPrice") or 0

    return chain, exp, spot


# ── Analysis ────────────────────────────────────────────────────────────

def analyze_sell_puts(ticker, spot, puts, exp, capital=None, discount_target=0.05):
    """Analyze sell put opportunities.

    Args:
        ticker: stock symbol
        spot: current price
        puts: DataFrame of put options
        exp: expiration date string
        capital: available cash (for position sizing)
        discount_target: desired OTM percentage (default 5%)
    """
    dte = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days
    if dte <= 0:
        return []

    results = []
    for _, r in puts.iterrows():
        strike = r['strike']
        if strike > spot * 1.02:  # skip ITM
            continue
        if strike < spot * 0.7:  # skip too far OTM
            continue

        mid = (r['bid'] + r['ask']) / 2 if r['bid'] > 0 else r['lastPrice']
        if mid <= 0:
            continue

        breakeven = strike - mid
        discount = (spot - strike) / spot * 100
        annualized = mid / strike * (365 / dte) * 100
        collateral = strike * 100  # cash-secured

        # Margin of safety
        margin = (spot - breakeven) / spot * 100

        results.append({
            "strike": strike,
            "premium": mid,
            "breakeven": breakeven,
            "discount_pct": discount,
            "annualized_pct": annualized,
            "margin_safety_pct": margin,
            "collateral": collateral,
            "volume": int(r.get('volume') or 0),
            "open_interest": int(r.get('openInterest') or 0),
            "iv": r.get('impliedVolatility', 0) or 0,
            "dte": dte,
        })

    # Sort by proximity to target discount
    results.sort(key=lambda x: abs(x['discount_pct'] - discount_target * 100))
    return results


def analyze_covered_calls(ticker, spot, calls, exp, shares=0, upside_target=0.05):
    """Analyze covered call opportunities."""
    dte = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days
    if dte <= 0:
        return []

    results = []
    for _, r in calls.iterrows():
        strike = r['strike']
        if strike < spot * 0.98:  # skip ITM
            continue
        if strike > spot * 1.3:  # skip too far OTM
            continue

        mid = (r['bid'] + r['ask']) / 2 if r['bid'] > 0 else r['lastPrice']
        if mid <= 0:
            continue

        upside = (strike - spot) / spot * 100
        annualized = mid / spot * (365 / dte) * 100
        total_return = ((strike - spot) + mid) / spot * 100  # cap gain + premium
        contracts = shares // 100

        results.append({
            "strike": strike,
            "premium": mid,
            "upside_pct": upside,
            "annualized_pct": annualized,
            "total_return_pct": total_return,
            "premium_income": mid * 100 * contracts,
            "contracts_possible": contracts,
            "volume": int(r.get('volume') or 0),
            "open_interest": int(r.get('openInterest') or 0),
            "iv": r.get('impliedVolatility', 0) or 0,
            "dte": dte,
        })

    results.sort(key=lambda x: abs(x['upside_pct'] - upside_target * 100))
    return results


# ── Rendering ───────────────────────────────────────────────────────────

def render_sell_puts(ticker, spot, results, top_n=5):
    if not results:
        return f"\n  {C.DIM}No sell put opportunities for {ticker}{C.RST}"

    lines = []
    lines.append(f"\n  {C.BOLD}{C.CYN}Sell Put Recommendations — {ticker} @ ${spot:.2f}{C.RST}")
    lines.append(f"  {'─' * 72}")
    lines.append(f"  {'Strike':>7}  {'Prem':>6}  {'BE':>8}  {'Disc%':>6}  {'Ann%':>6}  {'Safety':>7}  {'Collateral':>11}  {'Vol/OI':>10}")
    lines.append(f"  {'─' * 72}")

    for i, r in enumerate(results[:top_n]):
        strike = r['strike']
        prem = r['premium']
        be = r['breakeven']
        disc = r['discount_pct']
        ann = r['annualized_pct']
        safety = r['margin_safety_pct']
        coll = r['collateral']
        vol = r['volume']
        oi = r['open_interest']

        # Color code by annualized return
        ann_color = C.BGRN if ann > 30 else (C.BYEL if ann > 15 else C.WHT)
        safety_color = C.GRN if safety > 8 else (C.YEL if safety > 5 else C.RED)

        marker = " <<<" if i == 0 else ""
        lines.append(
            f"  {C.BOLD}${strike:>6.0f}{C.RST}  ${prem:>5.2f}  ${be:>7.2f}  "
            f"{disc:>5.1f}%  {ann_color}{ann:>5.1f}%{C.RST}  "
            f"{safety_color}{safety:>6.1f}%{C.RST}  "
            f"${coll:>9,.0f}  {vol:>5}/{oi:<5}{marker}"
        )

    lines.append(f"  {C.DIM}{'─' * 72}{C.RST}")
    lines.append(f"  {C.DIM}BE = breakeven | Disc% = OTM discount | Ann% = annualized return | Safety = margin from spot{C.RST}")

    return "\n".join(lines)


def render_covered_calls(ticker, spot, results, shares=0, top_n=5):
    if not results:
        return f"\n  {C.DIM}No covered call opportunities for {ticker}{C.RST}"

    lines = []
    contracts = shares // 100
    lines.append(f"\n  {C.BOLD}{C.CYN}Covered Call Recommendations — {ticker} @ ${spot:.2f} ({shares} shares = {contracts} contracts){C.RST}")
    lines.append(f"  {'─' * 72}")
    lines.append(f"  {'Strike':>7}  {'Prem':>6}  {'Up%':>6}  {'Ann%':>6}  {'Total%':>7}  {'Income':>8}  {'Vol/OI':>10}")
    lines.append(f"  {'─' * 72}")

    for i, r in enumerate(results[:top_n]):
        strike = r['strike']
        prem = r['premium']
        up = r['upside_pct']
        ann = r['annualized_pct']
        total = r['total_return_pct']
        income = r['premium_income']
        vol = r['volume']
        oi = r['open_interest']

        ann_color = C.BGRN if ann > 20 else (C.BYEL if ann > 10 else C.WHT)
        total_color = C.GRN if total > 5 else C.WHT

        marker = " <<<" if i == 0 else ""
        lines.append(
            f"  {C.BOLD}${strike:>6.0f}{C.RST}  ${prem:>5.2f}  {up:>5.1f}%  "
            f"{ann_color}{ann:>5.1f}%{C.RST}  "
            f"{total_color}{total:>6.1f}%{C.RST}  "
            f"${income:>7,.0f}  {vol:>5}/{oi:<5}{marker}"
        )

    lines.append(f"  {C.DIM}{'─' * 72}{C.RST}")
    lines.append(f"  {C.DIM}Up% = upside to strike | Ann% = annualized premium | Total% = cap gain + premium | Income = premium × 100 × contracts{C.RST}")

    return "\n".join(lines)


def render_wheel_analysis(ticker, spot, sell_puts, covered_calls, dte):
    """Render wheel strategy summary."""
    lines = []
    lines.append(f"\n  {C.BOLD}{C.MAG}Wheel Strategy — {ticker}{C.RST}")
    lines.append(f"  {'─' * 72}")

    best_sp = sell_puts[0] if sell_puts else None
    best_cc = covered_calls[0] if covered_calls else None

    if best_sp:
        lines.append(f"  {C.BOLD}Step 1: Sell Put{C.RST}")
        lines.append(f"    Sell ${best_sp['strike']:.0f} put → collect ${best_sp['premium']:.2f} premium")
        lines.append(f"    Breakeven: ${best_sp['breakeven']:.2f} ({best_sp['discount_pct']:.1f}% below spot)")
        lines.append(f"    Annualized: {C.BGRN}{best_sp['annualized_pct']:.1f}%{C.RST}")
        lines.append(f"    If assigned: cost basis = ${best_sp['breakeven']:.2f} (vs ${spot:.2f} current)")

    if best_cc:
        lines.append(f"\n  {C.BOLD}Step 2: Covered Call (if holding){C.RST}")
        lines.append(f"    Sell ${best_cc['strike']:.0f} call → collect ${best_cc['premium']:.2f} premium")
        lines.append(f"    Upside capture: {best_cc['upside_pct']:.1f}% to ${best_cc['strike']:.0f}")
        lines.append(f"    Annualized: {C.BGRN}{best_cc['annualized_pct']:.1f}%{C.RST}")

    if best_sp and best_cc:
        combined_ann = (best_sp['premium'] + best_cc['premium']) / spot * (365 / (dte * 2)) * 100
        lines.append(f"\n  {C.BOLD}Combined Wheel Annualized: {C.BGRN}{combined_ann:.1f}%{C.RST}")

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────

def analyze_ticker(ticker, shares=0, capital=0, mode="all"):
    """Run full options analysis for a single ticker."""
    chain, exp, spot = get_options_chain(ticker)
    if chain is None:
        print(f"  {C.RED}No options data for {ticker}{C.RST}")
        return

    dte = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days

    print(f"\n{C.BOLD}{'━' * 74}{C.RST}")
    print(f"{C.BOLD}{C.CYN}  OPTIONS ANALYSIS — {ticker}{C.RST}  {C.DIM}(spot ${spot:.2f} | exp {exp} | {dte} DTE){C.RST}")

    sell_puts = analyze_sell_puts(ticker, spot, chain.puts, exp, capital=capital)
    covered_calls = analyze_covered_calls(ticker, spot, chain.calls, exp, shares=shares)

    if mode in ("all", "sell-put"):
        print(render_sell_puts(ticker, spot, sell_puts))

    if mode in ("all", "covered-call"):
        print(render_covered_calls(ticker, spot, covered_calls, shares))

    if mode == "all":
        print(render_wheel_analysis(ticker, spot, sell_puts, covered_calls, dte))

    # Quick summary
    print(f"\n  {C.DIM}{'─' * 72}{C.RST}")
    if sell_puts:
        best = sell_puts[0]
        print(f"  {C.BOLD}Best Sell Put:{C.RST} ${best['strike']:.0f} → ${best['premium']:.2f} ({best['annualized_pct']:.1f}% ann) | BE ${best['breakeven']:.2f} ({best['margin_safety_pct']:.1f}% safety)")
    if covered_calls and shares >= 100:
        best = covered_calls[0]
        print(f"  {C.BOLD}Best Covered Call:{C.RST} ${best['strike']:.0f} → ${best['premium']:.2f} ({best['annualized_pct']:.1f}% ann) | Income ${best['premium_income']:,.0f}")


def main():
    parser = argparse.ArgumentParser(description="Options Analysis Agent")
    parser.add_argument("tickers", nargs="*", help="Ticker symbols (default: portfolio holdings)")
    parser.add_argument("--sell-put", action="store_true", help="Show sell put recommendations")
    parser.add_argument("--covered-call", action="store_true", help="Show covered call recommendations")
    parser.add_argument("--all", action="store_true", help="Full analysis (default)")
    parser.add_argument("--portfolio", action="store_true", help="Analyze all portfolio holdings")
    args = parser.parse_args()

    mode = "all"
    if args.sell_put and not args.covered_call:
        mode = "sell-put"
    elif args.covered_call and not args.sell_put:
        mode = "covered-call"

    # Determine tickers and shares
    tickers = args.tickers
    portfolio = load_portfolio()

    if not tickers:
        if portfolio:
            tickers = [p["ticker"] for p in portfolio.get("positions", [])]
        else:
            print(f"{C.RED}No tickers specified and no portfolio file found{C.RST}")
            return

    # Build position map
    pos_map = {}
    cash = 0
    if portfolio:
        for p in portfolio.get("positions", []):
            pos_map[p["ticker"]] = p.get("shares", 0)
        cash = portfolio.get("cash_usd", 0)

    print(f"\n{C.BOLD}{C.CYN}TRADING AGENTS — Options Analysis Agent{C.RST}")
    print(f"  {C.DIM}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.RST}")

    for ticker in tickers:
        shares = pos_map.get(ticker, 0)
        analyze_ticker(ticker, shares=shares, capital=cash, mode=mode)

    print(f"\n{C.BOLD}{'━' * 74}{C.RST}")
    print(f"  {C.DIM}Options Agent v1.0 | Weekend data: bid/ask may show 0, using lastPrice{C.RST}")


if __name__ == "__main__":
    main()
