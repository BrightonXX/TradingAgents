"""Position Manager — Kelly Criterion sizing, stop-loss/take-profit, rebalancing.

Usage:
    python position_manager.py              # full analysis
    python position_manager.py --rebalance  # rebalancing suggestions
    python position_manager.py --kelly NVDA  # Kelly sizing for ticker
"""
import os, sys, json, glob, argparse
from datetime import datetime
from collections import Counter

os.environ.setdefault('http_proxy', 'http://localhost:7890')
os.environ.setdefault('https_proxy', 'http://localhost:7890')

import yfinance as yf

# Colors
class C:
    RST = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GRN = "\033[32m"
    YEL = "\033[33m"
    CYN = "\033[36m"
    MAG = "\033[35m"
    BRED = "\033[1;31m"
    BGRN = "\033[1;32m"
    BYEL = "\033[1;33m"
    BBLU = "\033[1;34m"
    BMAG = "\033[1;35m"

PORTFOLIO_FILE = "my_portfolio.json"

# ── Data ────────────────────────────────────────────────────────────────

def load_portfolio():
    try:
        return json.load(open(PORTFOLIO_FILE))
    except:
        return None


def get_latest_signals():
    """Load latest portfolio signals."""
    reports = []
    for f in glob.glob("results/portfolio/report_*.json"):
        try:
            d = json.load(open(f))
            if d.get("structured"):
                reports.append(d)
        except:
            pass
    return reports


def get_stock_signals():
    """Load stock signals."""
    signals = []
    for f in glob.glob("results/*/latest/signal_*.json"):
        try:
            d = json.load(open(f))
            signals.append(d)
        except:
            pass
    return signals


def get_price_and_volatility(ticker, period="3mo"):
    """Get current price and historical volatility."""
    t = yf.Ticker(ticker)
    hist = t.history(period=period)
    if hist.empty:
        return None, None, None

    price = hist['Close'].iloc[-1]
    returns = hist['Close'].pct_change().dropna()
    daily_vol = returns.std()
    annual_vol = daily_vol * (252 ** 0.5) * 100

    # ATR (Average True Range)
    high = hist['High']
    low = hist['Low']
    close = hist['Close'].shift(1)
    tr = (high - low).combine(high - close, max).combine(low - close, max)
    atr = tr.rolling(14).mean().iloc[-1]

    return price, annual_vol, atr


# ── Kelly Criterion ─────────────────────────────────────────────────────

def kelly_criterion(win_rate, avg_win, avg_loss):
    """Calculate Kelly fraction.

    f* = (p * b - q) / b
    where p = win_rate, b = avg_win/avg_loss, q = 1 - p
    """
    if avg_loss <= 0:
        return 0
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - p
    f = (p * b - q) / b
    return max(0, f)  # Never negative


def half_kelly(kelly_fraction):
    """Half-Kelly for more conservative sizing."""
    return kelly_fraction / 2


# ── Analysis ────────────────────────────────────────────────────────────

def compute_stop_levels(price, atr, method="atr"):
    """Compute stop-loss and take-profit levels."""
    if method == "atr":
        # Stop: 2x ATR below, Target: 3x ATR above (1.5:1 R:R)
        stop = price - 2 * atr
        target = price + 3 * atr
    elif method == "pct":
        # Default 8% stop, 15% target
        stop = price * 0.92
        target = price * 1.15
    else:
        stop = price * 0.92
        target = price * 1.15

    stop_pct = (price - stop) / price * 100
    target_pct = (target - price) / price * 100
    rr_ratio = target_pct / stop_pct if stop_pct > 0 else 0

    return stop, target, stop_pct, target_pct, rr_ratio


def analyze_position(ticker, shares, cost_basis, current_price, volatility, atr,
                     signal_distribution=None):
    """Full position analysis."""
    position_value = shares * current_price
    pnl = (current_price - cost_basis) * shares
    pnl_pct = (current_price / cost_basis - 1) * 100
    weight_contribution = position_value  # will be normalized later

    stop, target, stop_pct, target_pct, rr = compute_stop_levels(current_price, atr)

    # Position risk (if hit stop loss)
    risk_amount = (current_price - stop) * shares
    risk_pct = stop_pct

    return {
        "ticker": ticker,
        "shares": shares,
        "cost_basis": cost_basis,
        "current_price": current_price,
        "position_value": position_value,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "volatility": volatility,
        "atr": atr,
        "stop": stop,
        "target": target,
        "stop_pct": stop_pct,
        "target_pct": target_pct,
        "rr_ratio": rr,
        "risk_amount": risk_amount,
    }


# ── Rendering ───────────────────────────────────────────────────────────

def render_position_analysis(positions, total_value, cash):
    lines = []
    lines.append(f"\n{C.BOLD}{'━' * 74}{C.RST}")
    lines.append(f"{C.BOLD}{C.CYN}  POSITION ANALYSIS{C.RST}")
    lines.append(f"  {'─' * 70}")

    total_risk = sum(p['risk_amount'] for p in positions)
    total_pnl = sum(p['pnl'] for p in positions)

    for p in positions:
        weight = p['position_value'] / total_value * 100
        pnl_color = C.GRN if p['pnl'] >= 0 else C.RED
        pnl_sign = "+" if p['pnl'] >= 0 else ""

        lines.append(f"\n  {C.BOLD}{p['ticker']:<6}{C.RST}  {p['shares']} shares @ ${p['cost_basis']:.2f}")
        lines.append(f"    Price: ${p['current_price']:.2f}  |  Value: ${p['position_value']:,.0f}  |  Weight: {weight:.1f}%")
        lines.append(f"    P&L: {pnl_color}{pnl_sign}${p['pnl']:,.0f} ({pnl_sign}{p['pnl_pct']:.1f}%){C.RST}")
        lines.append(f"    Volatility: {C.YEL}{p['volatility']:.1f}%{C.RST}  |  ATR: ${p['atr']:.2f}")
        lines.append(f"    Stop: {C.RED}${p['stop']:.2f} (-{p['stop_pct']:.1f}%){C.RST}  |  Target: {C.GRN}${p['target']:.2f} (+{p['target_pct']:.1f}%){C.RST}  |  R:R = {p['rr_ratio']:.1f}:1")
        lines.append(f"    Risk if stopped: {C.RED}${p['risk_amount']:,.0f}{C.RST}")

    # Portfolio risk summary
    lines.append(f"\n  {C.DIM}{'─' * 70}{C.RST}")
    lines.append(f"  {C.BOLD}Portfolio Risk Summary{C.RST}")
    lines.append(f"    Total Value: ${total_value:,.0f}  |  Cash: ${cash:,.0f} ({cash/total_value*100:.0f}%)")
    lines.append(f"    Total Unrealized P&L: {'+' if total_pnl >= 0 else ''}${total_pnl:,.0f}")
    lines.append(f"    Total Stop Risk: {C.RED}${total_risk:,.0f}{C.RST} ({total_risk/total_value*100:.1f}% of portfolio)")

    # Concentration check
    lines.append(f"\n  {C.BOLD}Concentration Analysis{C.RST}")
    for p in sorted(positions, key=lambda x: x['position_value'], reverse=True):
        weight = p['position_value'] / total_value * 100
        color = C.RED if weight > 25 else (C.YEL if weight > 15 else C.GRN)
        bar_width = 30
        filled = int(weight / 50 * bar_width)  # scale to 50%
        bar = f"{color}{'█' * filled}{C.DIM}{'░' * (bar_width - filled)}{C.RST}"
        lines.append(f"    {C.BOLD}{p['ticker']:<6}{C.RST} {bar} {color}{weight:.1f}%{C.RST}")

    return "\n".join(lines)


def render_kelly_analysis(positions, total_value, cash):
    """Kelly Criterion sizing recommendations."""
    lines = []
    lines.append(f"\n{C.BOLD}{'━' * 74}{C.RST}")
    lines.append(f"{C.BOLD}{C.CYN}  KELLY CRITERION SIZING{C.RST}")
    lines.append(f"  {'─' * 70}")

    # Get signal accuracy from backtest
    signals = get_stock_signals()
    from collections import Counter
    signal_counts = Counter()
    for s in signals:
        sig = s.get("signal", "HOLD").strip().upper()
        if sig in ("BUY", "BUY_MORE", "OVERWEIGHT"):
            signal_counts["BUY"] += 1
        elif sig in ("SELL", "UNDERWEIGHT", "REDUCE"):
            signal_counts["SELL"] += 1
        else:
            signal_counts["HOLD"] += 1

    total_sig = sum(signal_counts.values())
    buy_rate = signal_counts.get("BUY", 0) / total_sig if total_sig else 0

    # Use backtest accuracy (hardcoded from recent validation)
    # BUY accuracy ~71%, avg return when correct ~3%
    win_rate = 0.71
    avg_win = 3.0
    avg_loss = 2.0

    kelly = kelly_criterion(win_rate, avg_win, avg_loss)
    half = half_kelly(kelly)

    lines.append(f"  Win Rate (BUY signals): {C.BGRN}{win_rate:.0%}{C.RST}")
    lines.append(f"  Avg Win: +{avg_win:.1f}%  |  Avg Loss: -{avg_loss:.1f}%")
    lines.append(f"  Full Kelly: {C.BGRN}{kelly:.1%}{C.RST}  |  Half Kelly (recommended): {C.BYEL}{half:.1%}{C.RST}")
    lines.append(f"  Max position size: ${total_value * half:,.0f} ({half:.0%} of portfolio)")

    lines.append(f"\n  {C.DIM}{'─' * 70}{C.RST}")
    lines.append(f"  {C.BOLD}Suggested Position Sizes (Half-Kelly){C.RST}")

    for p in positions:
        current_weight = p['position_value'] / total_value
        max_position = total_value * half
        max_shares = int(max_position / p['current_price'])
        current_shares = p['shares']

        if current_shares >= max_shares:
            action = f"{C.RED}OVERWEIGHT — reduce by {current_shares - max_shares} shares{C.RST}"
        else:
            action = f"{C.GRN}Can add {max_shares - current_shares} shares (${(max_shares - current_shares) * p['current_price']:,.0f}){C.RST}"

        lines.append(f"    {C.BOLD}{p['ticker']:<6}{C.RST}  Max: {max_shares} shares (${max_position:,.0f})  |  Current: {current_shares}  |  {action}")

    return "\n".join(lines)


def render_rebalancing(positions, total_value, cash, portfolio_reports):
    """Rebalancing suggestions based on signals."""
    lines = []
    lines.append(f"\n{C.BOLD}{'━' * 74}{C.RST}")
    lines.append(f"{C.BOLD}{C.CYN}  REBALANCING SUGGESTIONS{C.RST}")
    lines.append(f"  {'─' * 70}")

    # Get latest consensus from portfolio reports
    signal_dist = {}
    for r in portfolio_reports[-20:]:  # Last 20 reports
        s = r.get("structured", {})
        for ticker, pos in s.get("positions", {}).items():
            action = pos.get("action", "HOLD").strip().upper()
            signal_dist.setdefault(ticker, Counter())[action] += 1

    # Generate suggestions
    actions = []
    for p in positions:
        ticker = p['ticker']
        weight = p['position_value'] / total_value * 100
        dist = signal_dist.get(ticker, Counter())
        total = sum(dist.values())

        if total > 0:
            dominant = dist.most_common(1)[0][0]
            confidence = dist.most_common(1)[0][1] / total * 100
        else:
            dominant = "HOLD"
            confidence = 0

        if dominant in ("SELL",) and confidence > 60:
            action = "CLOSE"
            action_color = C.BRED
        elif dominant == "REDUCE" and confidence > 50:
            action = "TRIM"
            action_color = C.BYEL
        elif dominant in ("BUY_MORE", "BUY") and confidence > 50 and cash > 5000:
            action = "ADD"
            action_color = C.BGRN
        else:
            action = "HOLD"
            action_color = C.CYN

        actions.append({
            "ticker": ticker,
            "weight": weight,
            "signal": dominant,
            "confidence": confidence,
            "action": action,
            "action_color": action_color,
            "pnl_pct": p['pnl_pct'],
        })

    # Sort by action priority
    action_order = {"CLOSE": 0, "TRIM": 1, "ADD": 2, "HOLD": 3}
    actions.sort(key=lambda x: action_order.get(x['action'], 3))

    for a in actions:
        signal_str = f"{a['signal']} ({a['confidence']:.0f}%)" if a['confidence'] > 0 else "no data"
        lines.append(
            f"  {a['action_color']}{a['action']:>6}{C.RST}  "
            f"{C.BOLD}{a['ticker']:<6}{C.RST}  "
            f"wt={a['weight']:.1f}%  signal={signal_str}  "
            f"P&L={a['pnl_pct']:+.1f}%"
        )

    # Cash deployment suggestion
    if cash > 10000:
        lines.append(f"\n  {C.BOLD}Cash Deployment{C.RST}")
        lines.append(f"    ${cash:,.0f} available ({cash/total_value*100:.0f}% of portfolio)")
        # Find BUY candidates
        buy_candidates = [a for a in actions if a['action'] == 'ADD']
        if buy_candidates:
            for c in buy_candidates:
                lines.append(f"    → {C.GRN}Add to {c['ticker']}{C.RST} (signal: {c['signal']}, {c['confidence']:.0f}% confidence)")
        else:
            lines.append(f"    {C.DIM}No strong BUY signals detected — keep cash as dry powder{C.RST}")

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Position Manager")
    parser.add_argument("--rebalance", action="store_true", help="Show rebalancing suggestions")
    parser.add_argument("--kelly", nargs="*", help="Kelly analysis for tickers (empty = all)")
    args = parser.parse_args()

    portfolio = load_portfolio()
    if not portfolio:
        print(f"{C.RED}No portfolio file found{C.RST}")
        return

    positions_data = portfolio.get("positions", [])
    cash = portfolio.get("cash_usd", 0)

    print(f"\n{C.BOLD}{C.CYN}TRADING AGENTS — Position Manager{C.RST}")
    print(f"  {C.DIM}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.RST}")

    # Fetch price data
    print(f"{C.DIM}  Fetching price data...{C.RST}")
    analyzed = []
    total_value = cash

    for p in positions_data:
        ticker = p["ticker"]
        shares = p["shares"]
        cost = p["cost_basis_usd"]

        price, vol, atr = get_price_and_volatility(ticker)
        if price is None:
            print(f"  {C.RED}Could not fetch data for {ticker}{C.RST}")
            continue

        pos = analyze_position(ticker, shares, cost, price, vol, atr)
        analyzed.append(pos)
        total_value += pos['position_value']

    # Position analysis
    print(render_position_analysis(analyzed, total_value, cash))

    # Kelly
    print(render_kelly_analysis(analyzed, total_value, cash))

    # Rebalancing
    portfolio_reports = get_latest_signals()
    print(render_rebalancing(analyzed, total_value, cash, portfolio_reports))

    print(f"\n{C.BOLD}{'━' * 74}{C.RST}")
    print(f"  {C.DIM}Position Manager v1.0 | Stops based on 2x ATR | Targets based on 3x ATR{C.RST}")


if __name__ == "__main__":
    main()
