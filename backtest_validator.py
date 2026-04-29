"""Backtest Validation — compare TradingAgents signals against actual price movements.

Usage:
    python backtest_validator.py               # validate all signals
    python backtest_validator.py --ticker NVDA  # single ticker
    python backtest_validator.py --horizon 5    # 5-day forward return
"""
import os, sys, json, glob, argparse
from datetime import datetime, timedelta
from collections import Counter, defaultdict

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
    BRED = "\033[1;31m"
    BGRN = "\033[1;32m"
    BYEL = "\033[1;33m"

SIGNAL_MAP = {
    "BUY": "BUY", "BUY_MORE": "BUY", "OVERWEIGHT": "BUY",
    "HOLD": "HOLD",
    "SELL": "SELL", "UNDERWEIGHT": "SELL", "REDUCE": "SELL",
}


def load_signals(ticker_filter=None):
    """Load all stock signals from results."""
    signals = []
    for f in glob.glob("results/*/latest/signal_*.json"):
        try:
            d = json.load(open(f))
            if ticker_filter and d.get("ticker") != ticker_filter:
                continue
            signals.append(d)
        except:
            pass
    return signals


def get_price_history(ticker):
    """Get daily price history."""
    t = yf.Ticker(ticker)
    hist = t.history(period="1mo")
    return {date.strftime("%Y-%m-%d"): row["Close"] for date, row in hist.iterrows()}


def compute_forward_return(prices, date, horizon=1):
    """Compute forward return from signal date."""
    # Signal date close → N trading days later close
    sorted_dates = sorted(prices.keys())
    if date not in sorted_dates:
        return None

    idx = sorted_dates.index(date)
    if idx + horizon >= len(sorted_dates):
        return None

    entry = prices[date]
    exit_ = prices[sorted_dates[idx + horizon]]
    return (exit_ - entry) / entry * 100


def classify_signal(signal_str):
    """Normalize signal to BUY/HOLD/SELL."""
    s = signal_str.strip().upper()
    return SIGNAL_MAP.get(s, "HOLD")


def evaluate_signal(signal, actual_return):
    """Evaluate if signal was correct.

    BUY + positive return = correct
    SELL + negative return = correct
    HOLD + small move = correct
    """
    if actual_return is None:
        return None

    if signal == "BUY":
        return actual_return > 0
    elif signal == "SELL":
        return actual_return < 0
    else:  # HOLD
        return abs(actual_return) < 3  # less than 3% move = HOLD was right


# ── Main ────────────────────────────────────────────────────────────────

def run_validation(ticker_filter=None, horizon=1):
    signals = load_signals(ticker_filter)
    if not signals:
        print(f"{C.RED}No signals found{C.RST}")
        return

    # Get unique tickers
    tickers = sorted(set(s["ticker"] for s in signals))

    # Fetch prices
    print(f"{C.BOLD}Fetching price data for {len(tickers)} tickers...{C.RST}")
    all_prices = {}
    for t in tickers:
        all_prices[t] = get_price_history(t)

    # Evaluate
    results = []  # (ticker, date, signal, return, correct)
    signal_stats = defaultdict(lambda: {"total": 0, "correct": 0, "returns": []})
    ticker_stats = defaultdict(lambda: {"total": 0, "correct": 0, "returns": []})

    for s in signals:
        ticker = s["ticker"]
        date = s["date"]
        raw_signal = s["signal"]
        signal = classify_signal(raw_signal)
        ret = compute_forward_return(all_prices.get(ticker, {}), date, horizon)

        if ret is None:
            continue

        correct = evaluate_signal(signal, ret)
        results.append({
            "ticker": ticker,
            "date": date,
            "signal": signal,
            "raw_signal": raw_signal,
            "return_pct": ret,
            "correct": correct,
        })

        signal_stats[signal]["total"] += 1
        if correct:
            signal_stats[signal]["correct"] += 1
        signal_stats[signal]["returns"].append(ret)

        ticker_stats[ticker]["total"] += 1
        if correct:
            ticker_stats[ticker]["correct"] += 1
        ticker_stats[ticker]["returns"].append(ret)

    # ── Render ──────────────────────────────────────────────────────────

    print(f"\n{C.BOLD}{'━' * 74}{C.RST}")
    print(f"{C.BOLD}{C.CYN}  BACKTEST VALIDATION — Signal vs Reality ({horizon}-day forward){C.RST}")
    print(f"  {C.DIM}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {len(results)} signals evaluated{C.RST}")
    print(f"{C.BOLD}{'━' * 74}{C.RST}")

    # Individual signals
    print(f"\n{C.BOLD}  Detailed Results{C.RST}")
    print(f"  {'─' * 70}")
    print(f"  {'Ticker':>6}  {'Date':>10}  {'Signal':>6}  {'Return':>7}  {'Verdict':>8}")
    print(f"  {'─' * 70}")

    # Group by ticker+date, show consensus
    td_groups = defaultdict(list)
    for r in results:
        td_groups[(r["ticker"], r["date"])].append(r)

    for (ticker, date), group in sorted(td_groups.items()):
        signals_list = [r["signal"] for r in group]
        consensus = Counter(signals_list).most_common(1)[0][0]
        avg_ret = sum(r["return_pct"] for r in group) / len(group)
        correct = evaluate_signal(consensus, avg_ret)

        ret_color = C.GRN if avg_ret >= 0 else C.RED
        ret_sign = "+" if avg_ret >= 0 else ""
        verdict = "CORRECT" if correct else "WRONG"
        v_color = C.BGRN if correct else C.BRED

        n = len(group)
        sig_counts = Counter(signals_list)
        sig_str = "/".join(f"{sig}:{cnt}" for sig, cnt in sig_counts.most_common())

        print(
            f"  {C.BOLD}{ticker:>6}{C.RST}  {date:>10}  {consensus:>6}  "
            f"{ret_color}{ret_sign}{avg_ret:>6.2f}%{C.RST}  "
            f"{v_color}{verdict:>8}{C.RST}  {C.DIM}({n} runs: {sig_str}){C.RST}"
        )

    # Signal accuracy summary
    print(f"\n{C.BOLD}{'━' * 74}{C.RST}")
    print(f"{C.BOLD}{C.CYN}  Signal Accuracy{C.RST}")
    print(f"  {'─' * 70}")

    total_correct = sum(s["correct"] for s in signal_stats.values())
    total_signals = sum(s["total"] for s in signal_stats.values())
    overall_acc = total_correct / total_signals * 100 if total_signals else 0

    acc_color = C.BGRN if overall_acc > 60 else (C.BYEL if overall_acc > 50 else C.BRED)
    print(f"  Overall: {acc_color}{overall_acc:.0f}%{C.RST} ({total_correct}/{total_signals})")

    for sig in ["BUY", "HOLD", "SELL"]:
        stats = signal_stats[sig]
        if stats["total"] == 0:
            continue
        acc = stats["correct"] / stats["total"] * 100
        avg_ret = sum(stats["returns"]) / len(stats["returns"])
        acc_color = C.BGRN if acc > 60 else (C.BYEL if acc > 50 else C.BRED)
        ret_color = C.GRN if avg_ret >= 0 else C.RED
        ret_sign = "+" if avg_ret >= 0 else ""

        # Bar
        bar_width = 20
        filled = int(acc / 100 * bar_width)
        bar = f"{C.GRN}{'█' * filled}{C.DIM}{'░' * (bar_width - filled)}{C.RST}"

        print(
            f"  {C.BOLD}{sig:>6}{C.RST}  {bar} {acc_color}{acc:>3.0f}%{C.RST} "
            f"({stats['correct']}/{stats['total']})  "
            f"Avg return: {ret_color}{ret_sign}{avg_ret:.2f}%{C.RST}"
        )

    # Per-ticker summary
    print(f"\n{C.BOLD}{'━' * 74}{C.RST}")
    print(f"{C.BOLD}{C.CYN}  Per-Ticker Accuracy{C.RST}")
    print(f"  {'─' * 70}")

    for ticker in sorted(ticker_stats.keys()):
        stats = ticker_stats[ticker]
        if stats["total"] == 0:
            continue
        acc = stats["correct"] / stats["total"] * 100
        avg_ret = sum(stats["returns"]) / len(stats["returns"])
        acc_color = C.BGRN if acc > 60 else (C.BYEL if acc > 50 else C.BRED)
        ret_color = C.GRN if avg_ret >= 0 else C.RED
        ret_sign = "+" if avg_ret >= 0 else ""

        bar_width = 20
        filled = int(acc / 100 * bar_width)
        bar = f"{C.GRN}{'█' * filled}{C.DIM}{'░' * (bar_width - filled)}{C.RST}"

        print(
            f"  {C.BOLD}{ticker:>6}{C.RST}  {bar} {acc_color}{acc:>3.0f}%{C.RST} "
            f"({stats['correct']}/{stats['total']})  "
            f"Avg return: {ret_color}{ret_sign}{avg_ret:.2f}%{C.RST}"
        )

    print(f"\n{C.BOLD}{'━' * 74}{C.RST}")
    print(f"  {C.DIM}Backtest Validator v1.0 | HOLD correct if |return| < 3%{C.RST}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest Validation")
    parser.add_argument("--ticker", help="Filter by ticker")
    parser.add_argument("--horizon", type=int, default=1, help="Forward return horizon in trading days")
    args = parser.parse_args()
    run_validation(args.ticker, args.horizon)
