"""Update portfolio prices from yfinance.

Usage:
    python update_prices.py
"""
import os, json
os.environ.setdefault('http_proxy', 'http://localhost:7890')
os.environ.setdefault('https_proxy', 'http://localhost:7890')

import yfinance as yf

PORTFOLIO_FILE = "my_portfolio.json"

def main():
    with open(PORTFOLIO_FILE) as f:
        pf = json.load(f)

    total_value = pf.get("cash_usd", 0)
    updated = []

    for p in pf['positions']:
        t = yf.Ticker(p['ticker'])
        info = t.info or {}
        price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
        if price == 0:
            price = info.get('previousClose', 0)
        prev = info.get('regularMarketPreviousClose') or 0
        old = p['current_price_usd']

        p['current_price_usd'] = round(price, 2)
        chg = (price - prev) / prev * 100 if prev else 0
        pnl = (price - p['cost_basis_usd']) / p['cost_basis_usd'] * 100
        value = p['shares'] * price
        total_value += value

        sign = "+" if chg >= 0 else ""
        updated.append(f"  {p['ticker']:<6} ${old:>8.2f} -> ${price:>8.2f}  {sign}{chg:.2f}% (P&L: {sign}{pnl:.1f}%)  ${value:,.0f}")

    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(pf, f, indent=2, ensure_ascii=False)

    cash = pf.get("cash_usd", 0)
    print(f"Portfolio updated:\n" + "\n".join(updated))
    print(f"\n  Total: ${total_value:,.0f}  Cash: ${cash:,.0f} ({cash/total_value*100:.0f}%)")


if __name__ == "__main__":
    main()
