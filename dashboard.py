"""Signal Confidence Dashboard — aggregate TradingAgents signals into a terminal dashboard.

Usage:
    python dashboard.py              # full dashboard
    python dashboard.py --watch      # auto-refresh every 60s
    python dashboard.py --json       # output raw JSON
"""
import json, glob, os, sys, time
from datetime import datetime
from collections import Counter

# Optional: live prices via yfinance
try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

# ── Config ──────────────────────────────────────────────────────────────
PORTFOLIO_FILE = "my_portfolio.json"
RESULTS_DIR = "results"
PORTFOLIO_DIR = f"{RESULTS_DIR}/portfolio"

# Colors
class C:
    RST = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GRN = "\033[32m"
    YEL = "\033[33m"
    BLU = "\033[34m"
    MAG = "\033[35m"
    CYN = "\033[36m"
    WHT = "\033[37m"
    BRED = "\033[1;31m"
    BGRN = "\033[1;32m"
    BYEL = "\033[1;33m"
    BBLU = "\033[1;34m"
    BMAG = "\033[1;35m"
    BCYN = "\033[1;36m"

SIGNAL_COLORS = {
    "BUY_MORE": C.BGRN, "BUY": C.BGRN,
    "HOLD": C.BYEL,
    "REDUCE": C.YEL,
    "SELL": C.BRED,
}

SIGNAL_ORDER = ["BUY_MORE", "BUY", "HOLD", "REDUCE", "SELL"]
SIGNAL_SHORT = {"BUY_MORE": "BM", "BUY": "B", "HOLD": "H", "REDUCE": "R", "SELL": "S"}

WIDTH = 72

# ── Data Loading ────────────────────────────────────────────────────────

def load_stock_signals():
    """Load all stock signal JSONs."""
    signals = []
    for f in glob.glob(f"{RESULTS_DIR}/*/latest/signal_*.json"):
        try:
            d = json.load(open(f))
            signals.append(d)
        except:
            pass
    return signals


def load_portfolio_reports():
    """Load all portfolio report JSONs."""
    reports = []
    for f in glob.glob(f"{PORTFOLIO_DIR}/report_*.json"):
        try:
            d = json.load(open(f))
            if d.get("structured"):
                reports.append(d)
        except:
            pass
    return reports


def load_debate_reports():
    """Load debate-specific report JSONs."""
    debates = []
    for f in glob.glob(f"{PORTFOLIO_DIR}/debate_*.json"):
        try:
            debates.append(json.load(open(f)))
        except:
            pass
    return debates


def load_portfolio():
    """Load current portfolio from my_portfolio.json."""
    try:
        return json.load(open(PORTFOLIO_FILE))
    except:
        return None


# ── Analysis ────────────────────────────────────────────────────────────

def analyze_stock_distribution(signals):
    """Get signal distribution per ticker."""
    by_ticker = {}
    for s in signals:
        t = s.get("ticker", "?")
        sig = s.get("signal", "?").strip().upper()
        # Normalize
        if sig not in SIGNAL_ORDER:
            sig = "HOLD"
        by_ticker.setdefault(t, Counter())[sig] += 1
    return by_ticker


def analyze_portfolio_distribution(reports):
    """Get action distribution per position from portfolio reports."""
    by_ticker = {}
    scores = []
    total_cost = {"input_cny": 0, "output_cny": 0, "cache_cny": 0, "total_cny": 0}
    total_tokens = {"tokens_in": 0, "tokens_out": 0, "cache_read": 0, "llm_calls": 0}

    for r in reports:
        s = r.get("structured", {})
        if s.get("portfolio_score"):
            scores.append(s["portfolio_score"])
        for ticker, pos in s.get("positions", {}).items():
            action = pos.get("action", "HOLD").strip().upper()
            if action not in SIGNAL_ORDER:
                action = "HOLD"
            by_ticker.setdefault(ticker, Counter())[action] += 1
        # Aggregate cost
        cost = r.get("cost_cny", {})
        for k in total_cost:
            total_cost[k] += cost.get(k, 0)
        tok = r.get("token_usage", {})
        for k in total_tokens:
            total_tokens[k] += tok.get(k, 0)

    return by_ticker, scores, total_cost, total_tokens


def fetch_live_prices(tickers):
    """Fetch live/pre-market prices via yfinance. Returns dict: ticker -> {price, change_pct, premarket_price, premarket_change_pct}."""
    if not HAS_YF:
        return {}
    result = {}
    try:
        for ticker in tickers:
            t = yf.Ticker(ticker)
            info = t.info or {}
            price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
            prev = info.get("regularMarketPreviousClose") or info.get("previousClose") or 0
            pre = info.get("preMarketPrice") or 0
            pre_prev = info.get("preMarketPreviousClose") or prev

            change_pct = ((price - prev) / prev * 100) if prev else 0
            pre_change_pct = ((pre - pre_prev) / pre_prev * 100) if pre_prev and pre else 0

            result[ticker] = {
                "price": price,
                "prev": prev,
                "change_pct": change_pct,
                "premarket": pre,
                "premarket_change_pct": pre_change_pct,
            }
    except Exception as e:
        pass
    return result


# ── Rendering ───────────────────────────────────────────────────────────

def bar(count, total, width=20, fill="█", empty="░"):
    """Render a horizontal bar."""
    if total == 0:
        return empty * width
    filled = int(count / total * width)
    return fill * filled + empty * (width - filled)


def signal_bar(dist, width=30):
    """Render a multi-color signal distribution bar."""
    total = sum(dist.values())
    if total == 0:
        return C.DIM + "░" * width + C.RST

    parts = []
    remaining = width
    for sig in SIGNAL_ORDER:
        count = dist.get(sig, 0)
        if count == 0:
            continue
        bar_width = max(1, int(count / total * width))
        bar_width = min(bar_width, remaining)
        color = SIGNAL_COLORS.get(sig, C.WHT)
        parts.append(f"{color}{'█' * bar_width}{C.RST}")
        remaining -= bar_width
        if remaining <= 0:
            break

    if remaining > 0:
        parts.append(C.DIM + "░" * remaining + C.RST)

    return "".join(parts)


def format_pct(count, total):
    if total == 0:
        return "  0%"
    return f"{count/total*100:3.0f}%"


def render_header(portfolio_data):
    """Render dashboard header with portfolio summary."""
    lines = []
    lines.append(f"{C.BOLD}{'━' * WIDTH}{C.RST}")
    lines.append(f"{C.BOLD}{C.CYN}  TRADING AGENTS — Signal Confidence Dashboard{C.RST}")
    lines.append(f"{C.DIM}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.RST}")
    lines.append(f"{C.BOLD}{'━' * WIDTH}{C.RST}")

    if portfolio_data:
        total_value = portfolio_data.get("cash_usd", 0)
        positions = portfolio_data.get("positions", [])
        lines.append(f"\n{C.BOLD}  Portfolio Overview{C.RST}")
        lines.append(f"  {'─' * (WIDTH - 4)}")

        for p in positions:
            ticker = p["ticker"]
            shares = p["shares"]
            cost = p["cost_basis_usd"]
            price = p["current_price_usd"]
            value = shares * price
            pnl = (price - cost) * shares
            pnl_pct = (price / cost - 1) * 100
            total_value += value

            pnl_color = C.GRN if pnl >= 0 else C.RED
            pnl_sign = "+" if pnl >= 0 else ""
            lines.append(
                f"  {C.BOLD}{ticker:<6}{C.RST} {shares:>4} shares  "
                f"${price:>8.2f}  "
                f"{pnl_color}{pnl_sign}${abs(pnl):>7.0f} ({pnl_sign}{pnl_pct:.1f}%){C.RST}  "
                f"${value:>8,.0f}"
            )

        cash = portfolio_data.get("cash_usd", 0)
        cash_pct = cash / total_value * 100 if total_value else 0
        lines.append(f"  {C.DIM}{'─' * (WIDTH - 4)}{C.RST}")
        lines.append(
            f"  {C.BOLD}Cash{C.RST}  ${cash:>10,.0f} ({cash_pct:.0f}%)    "
            f"{C.BOLD}Total{C.RST} ${total_value:>10,.0f}"
        )

    return "\n".join(lines)


def render_market_data(live_prices, portfolio_data):
    """Render live/pre-market price section."""
    if not live_prices:
        return ""

    lines = []
    lines.append(f"\n{C.BOLD}{'━' * WIDTH}{C.RST}")
    lines.append(f"{C.BOLD}{C.CYN}  Market Data (Live){C.RST}")
    lines.append(f"  {'─' * (WIDTH - 4)}")

    for ticker in sorted(live_prices.keys()):
        d = live_prices[ticker]
        price = d.get("price", 0)
        chg = d.get("change_pct", 0)
        pre = d.get("premarket", 0)
        pre_chg = d.get("premarket_change_pct", 0)

        price_str = f"${price:>8.2f}" if price else f"{'─':>9}"
        chg_color = C.GRN if chg >= 0 else C.RED
        chg_str = f"{chg_color}{chg:>+6.2f}%{C.RST}" if price else ""

        pre_str = ""
        if pre:
            pre_color = C.GRN if pre_chg >= 0 else C.RED
            pre_str = f"  Pre: ${pre:>8.2f} {pre_color}{pre_chg:>+6.2f}%{C.RST}"

        lines.append(f"  {C.BOLD}{ticker:<6}{C.RST}  {price_str} {chg_str}{pre_str}")

    return "\n".join(lines)


def render_stock_signals(stock_dist):
    """Render stock signal distribution section."""
    lines = []
    lines.append(f"\n{C.BOLD}{'━' * WIDTH}{C.RST}")
    lines.append(f"{C.BOLD}{C.CYN}  Stock Analysis Signals{C.RST}")
    lines.append(f"  {'─' * (WIDTH - 4)}")

    for ticker in sorted(stock_dist.keys()):
        dist = stock_dist[ticker]
        total = sum(dist.values())
        sbar = signal_bar(dist, width=25)

        # Get dominant signal
        dominant = dist.most_common(1)[0][0] if dist else "HOLD"
        dom_color = SIGNAL_COLORS.get(dominant, C.WHT)

        # Build percentage string
        pcts = []
        for sig in SIGNAL_ORDER:
            c = dist.get(sig, 0)
            if c > 0:
                sc = SIGNAL_COLORS.get(sig, C.WHT)
                short = SIGNAL_SHORT[sig]
                pcts.append(f"{sc}{short}:{c}/{total}{C.RST}")

        pct_str = " ".join(pcts)
        lines.append(f"  {C.BOLD}{ticker:<6}{C.RST} {sbar}  {dom_color}{dominant:<8}{C.RST} {pct_str}")

    # Legend
    lines.append(f"  {C.DIM}{'─' * (WIDTH - 4)}{C.RST}")
    legend = "  ".join(f"{SIGNAL_COLORS[s]}{s}{C.RST}" for s in SIGNAL_ORDER)
    lines.append(f"  {legend}")

    return "\n".join(lines)


def render_portfolio_signals(portfolio_dist, scores, total_cost, total_tokens, n_reports):
    """Render portfolio signal distribution section."""
    lines = []
    lines.append(f"\n{C.BOLD}{'━' * WIDTH}{C.RST}")
    lines.append(f"{C.BOLD}{C.CYN}  Portfolio Analysis Signals ({n_reports} reports){C.RST}")
    lines.append(f"  {'─' * (WIDTH - 4)}")

    for ticker in sorted(portfolio_dist.keys()):
        dist = portfolio_dist[ticker]
        total = sum(dist.values())
        sbar = signal_bar(dist, width=25)

        dominant = dist.most_common(1)[0][0] if dist else "HOLD"
        dom_color = SIGNAL_COLORS.get(dominant, C.WHT)

        pcts = []
        for sig in SIGNAL_ORDER:
            c = dist.get(sig, 0)
            if c > 0:
                sc = SIGNAL_COLORS.get(sig, C.WHT)
                short = SIGNAL_SHORT[sig]
                pcts.append(f"{sc}{short}:{c}/{total}{C.RST}")

        pct_str = " ".join(pcts)
        lines.append(f"  {C.BOLD}{ticker:<6}{C.RST} {sbar}  {dom_color}{dominant:<8}{C.RST} {pct_str}")

    # Legend
    lines.append(f"  {C.DIM}{'─' * (WIDTH - 4)}{C.RST}")
    legend = "  ".join(f"{SIGNAL_COLORS[s]}{s}{C.RST}" for s in SIGNAL_ORDER)
    lines.append(f"  {legend}")

    # Score distribution
    if scores:
        avg_score = sum(scores) / len(scores)
        score_bar_min = 1
        score_bar_max = 10
        score_width = 30
        filled = int((avg_score - score_bar_min) / (score_bar_max - score_bar_min) * score_width)
        score_color = C.GRN if avg_score >= 7 else (C.YEL if avg_score >= 5 else C.RED)

        lines.append(f"\n  {C.BOLD}Portfolio Score{C.RST}")
        lines.append(f"  Avg: {score_color}{avg_score:.1f}/10{C.RST}")
        lines.append(f"  {score_color}{'█' * filled}{C.DIM}{'░' * (score_width - filled)}{C.RST}")

        # Score distribution
        score_counts = Counter(round(s) for s in scores)
        dist_str = "  ".join(f"{C.CYN}{k}:{C.RST}{v}" for k, v in sorted(score_counts.items()))
        lines.append(f"  {C.DIM}Distribution: {dist_str}{C.RST}")

    # Cost summary
    tc = total_cost
    lines.append(f"\n  {C.BOLD}Cost Summary{C.RST}")
    lines.append(f"  Total runs: {n_reports}")
    lines.append(f"  Total cost: ¥{tc['total_cny']:.2f} (in: ¥{tc['input_cny']:.2f} / out: ¥{tc['output_cny']:.2f} / cache: ¥{tc['cache_cny']:.2f})")
    tt = total_tokens
    lines.append(f"  Tokens: {tt['tokens_in']//1000}k in / {tt['tokens_out']//1000}k out / {tt['cache_read']//1000}k cache ({tt['llm_calls']} calls)")
    if n_reports > 0:
        lines.append(f"  Avg cost/run: ¥{tc['total_cny']/n_reports:.2f}")

    return "\n".join(lines)


def render_latest_signals(reports, n=3):
    """Show latest N portfolio report signals."""
    lines = []
    lines.append(f"\n{C.BOLD}{'━' * WIDTH}{C.RST}")
    lines.append(f"{C.BOLD}{C.CYN}  Latest Portfolio Signals{C.RST}")
    lines.append(f"  {'─' * (WIDTH - 4)}")

    # Sort by file modification time
    latest = sorted(reports, key=lambda r: r.get("date", ""), reverse=True)[:n]

    for r in latest:
        s = r.get("structured", {})
        if not s:
            continue
        date = r.get("date", "?")
        score = s.get("portfolio_score", "?")
        score_color = C.GRN if isinstance(score, (int, float)) and score >= 7 else (C.YEL if isinstance(score, (int, float)) and score >= 5 else C.RED)
        cost = r.get("cost_cny", {})
        cost_val = cost.get("total_cny", 0)

        lines.append(f"\n  {C.DIM}{date}{C.RST}  Score: {score_color}{score}/10{C.RST}  Cost: ¥{cost_val:.2f}")

        for ticker, pos in s.get("positions", {}).items():
            action = pos.get("action", "?")
            confidence = pos.get("confidence", "?")
            target = pos.get("target_price", "?")
            reason = pos.get("reason", "")[:60]
            ac = SIGNAL_COLORS.get(action, C.WHT)
            lines.append(f"    {C.BOLD}{ticker:<6}{C.RST} {ac}{action:<8}{C.RST} conf={confidence} tgt=${target}  {C.DIM}{reason}{C.RST}")

    return "\n".join(lines)


def render_consensus_matrix(portfolio_dist, stock_dist):
    """Render consensus matrix comparing stock vs portfolio signals."""
    lines = []
    lines.append(f"\n{C.BOLD}{'━' * WIDTH}{C.RST}")
    lines.append(f"{C.BOLD}{C.CYN}  Consensus Matrix (Stock vs Portfolio){C.RST}")
    lines.append(f"  {'─' * (WIDTH - 4)}")

    all_tickers = sorted(set(list(stock_dist.keys()) + list(portfolio_dist.keys())))

    for ticker in all_tickers:
        s_dist = stock_dist.get(ticker, Counter())
        p_dist = portfolio_dist.get(ticker, Counter())

        s_dom = s_dist.most_common(1)[0][0] if s_dist else "—"
        p_dom = p_dist.most_common(1)[0][0] if p_dist else "—"

        s_total = sum(s_dist.values())
        p_total = sum(p_dist.values())

        s_color = SIGNAL_COLORS.get(s_dom, C.DIM)
        p_color = SIGNAL_COLORS.get(p_dom, C.DIM)

        # Agreement indicator
        agree = "✓" if s_dom == p_dom else ("~" if (s_dom in ("HOLD", "REDUCE") and p_dom in ("HOLD", "REDUCE")) or (s_dom in ("BUY", "BUY_MORE") and p_dom in ("BUY", "BUY_MORE")) or (s_dom == "SELL" and p_dom == "SELL") else "✗")
        agree_color = C.GRN if agree == "✓" else (C.YEL if agree == "~" else C.RED)

        lines.append(
            f"  {C.BOLD}{ticker:<6}{C.RST}  "
            f"Stock({s_total:>2}): {s_color}{s_dom:<8}{C.RST}  "
            f"Portfolio({p_total:>3}): {p_color}{p_dom:<8}{C.RST}  "
            f"{agree_color}{agree}{C.RST}"
        )

    return "\n".join(lines)


def render_footer():
    lines = []
    lines.append(f"\n{C.BOLD}{'━' * WIDTH}{C.RST}")
    lines.append(f"{C.DIM}  TradingAgents Dashboard v1.0{C.RST}")
    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────

def run_dashboard():
    """Build and display the full dashboard."""
    # Load data
    portfolio_data = load_portfolio()
    stock_signals = load_stock_signals()
    portfolio_reports = load_portfolio_reports()

    # Analyze
    stock_dist = analyze_stock_distribution(stock_signals)
    portfolio_dist, scores, total_cost, total_tokens = analyze_portfolio_distribution(portfolio_reports)

    # Fetch live prices for portfolio tickers
    live_prices = {}
    if HAS_YF and portfolio_data:
        tickers = [p["ticker"] for p in portfolio_data.get("positions", [])]
        live_prices = fetch_live_prices(tickers)

    # Render
    output = []
    output.append(render_header(portfolio_data))
    output.append(render_market_data(live_prices, portfolio_data))
    output.append(render_consensus_matrix(portfolio_dist, stock_dist))
    output.append(render_stock_signals(stock_dist))
    output.append(render_portfolio_signals(portfolio_dist, scores, total_cost, total_tokens, len(portfolio_reports)))
    output.append(render_latest_signals(portfolio_reports, n=5))
    output.append(render_footer())

    # Clear screen and print
    print("\033[2J\033[H", end="")  # clear screen
    print("\n".join(output))


def run_dashboard_json():
    """Output dashboard data as JSON."""
    stock_signals = load_stock_signals()
    portfolio_reports = load_portfolio_reports()

    stock_dist = analyze_stock_distribution(stock_signals)
    portfolio_dist, scores, total_cost, total_tokens = analyze_portfolio_distribution(portfolio_reports)

    # Convert Counters to plain dicts
    stock_out = {t: dict(d) for t, d in stock_dist.items()}
    portfolio_out = {t: dict(d) for t, d in portfolio_dist.items()}

    result = {
        "timestamp": datetime.now().isoformat(),
        "stock_signals": {"count": len(stock_signals), "distribution": stock_out},
        "portfolio_signals": {
            "count": len(portfolio_reports),
            "distribution": portfolio_out,
            "scores": {"mean": sum(scores)/len(scores) if scores else 0, "values": scores},
            "cost": total_cost,
            "tokens": total_tokens,
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--json" in args:
        run_dashboard_json()
    elif "--watch" in args:
        try:
            while True:
                run_dashboard()
                print(f"\n{C.DIM}  Refreshing in 60s... (Ctrl+C to exit){C.RST}")
                time.sleep(60)
        except KeyboardInterrupt:
            print(f"\n{C.DIM}  Dashboard stopped.{C.RST}")
    else:
        run_dashboard()
