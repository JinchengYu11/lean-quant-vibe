#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import requests
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Alpaca Portfolio Rebalancing Script")
    parser.add_argument(
        "--target",
        type=str,
        default="target_portfolio.json",
        help="Path to the target portfolio JSON file (default: target_portfolio.json)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate trades but do not place orders (or set DRY_RUN=1)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use Live Trading API instead of Paper Trading"
    )
    return parser.parse_args()

def load_env():
    """Load environment variables from a local .env file if it exists."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    # Strip quotes if present
                    val = val.strip().strip("'").strip('"')
                    os.environ[key.strip()] = val
        print("Loaded environment variables from .env file.")

def get_credentials(use_live, dry_run):
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    is_mocked = False
    
    if not api_key or not secret_key:
        if dry_run:
            print("Warning: ALPACA_API_KEY and ALPACA_SECRET_KEY not set. Using MOCK data for dry-run simulation.")
            api_key = "MOCK_KEY"
            secret_key = "MOCK_SECRET"
            base_url = "https://paper-api.alpaca.markets"
            is_mocked = True
        else:
            print("Error: ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables must be set.")
            sys.exit(1)
    else:
        if use_live:
            base_url = "https://api.alpaca.markets"
            print("Using Alpaca LIVE Trading API.")
        else:
            base_url = "https://paper-api.alpaca.markets"
            print("Using Alpaca PAPER Trading API.")
            
    return api_key, secret_key, base_url, is_mocked

def get_headers(api_key, secret_key):
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
        "Content-Type": "application/json"
    }

def get_account_info(base_url, headers):
    url = f"{base_url}/v2/account"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching account info (HTTP {response.status_code}): {response.text}")
        sys.exit(1)
    return response.json()

def get_current_positions(base_url, headers):
    url = f"{base_url}/v2/positions"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching positions (HTTP {response.status_code}): {response.text}")
        sys.exit(1)
    return response.json()

def get_latest_price(symbol, headers):
    """
    Fetch the latest trade price from Alpaca Market Data API.
    Used for estimating order quantities in dry-run or logs.
    """
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return float(data.get("trade", {}).get("p", 0))
    except Exception as e:
        pass
    return None

def place_order(base_url, headers, symbol, side, qty=None, notional=None):
    url = f"{base_url}/v2/orders"
    payload = {
        "symbol": symbol,
        "side": side,
        "type": "market",
        "time_in_force": "day"
    }
    
    if qty is not None:
        payload["qty"] = str(qty)
    elif notional is not None:
        payload["notional"] = str(round(notional, 2))
    else:
        raise ValueError("Either qty or notional must be specified.")
        
    response = requests.post(url, headers=headers, json=payload)
    return response

def main():
    # Load env first
    load_env()
    
    args = parse_args()
    
    # Check dry run
    dry_run = args.dry_run or os.getenv("DRY_RUN") == "1"
    
    # Get credentials
    api_key, secret_key, base_url, is_mocked = get_credentials(args.live, dry_run)
    headers = get_headers(api_key, secret_key)
    
    # 1. Load target portfolio
    if not os.path.exists(args.target):
        print(f"Error: Target portfolio file '{args.target}' not found.")
        sys.exit(1)
        
    with open(args.target, "r", encoding="utf-8") as f:
        target_portfolio = json.load(f)
        
    print(f"Loaded target portfolio with {len(target_portfolio)} assets.")
    
    # Validate target weights
    total_target_weight = sum(target_portfolio.values())
    print(f"Total target weight: {total_target_weight:.2%}")
    if total_target_weight > 1.001:
        print("Warning: Total target weight exceeds 100%. Scaling weights down to 99% total.")
        scale_factor = 0.99 / total_target_weight
        target_portfolio = {k: v * scale_factor for k, v in target_portfolio.items()}
        total_target_weight = sum(target_portfolio.values())
        print(f"Scaled total target weight: {total_target_weight:.2%}")
        
    # 2. Get Account Info
    if is_mocked:
        equity = 100000.00
        cash = 45000.00
        print(f"[MOCK] Account Equity: ${equity:,.2f} | Cash: ${cash:,.2f}")
    else:
        account = get_account_info(base_url, headers)
        equity = float(account["equity"])
        cash = float(account["cash"])
        print(f"Account Equity: ${equity:,.2f} | Cash: ${cash:,.2f}")
    
    # 3. Get Current Positions
    current_positions = {}
    if is_mocked:
        positions_raw = [
            {"symbol": "SPY", "qty": "100.0", "market_value": "45000.00", "current_price": "450.00"},
            {"symbol": "TSLA", "qty": "50.0", "market_value": "10000.00", "current_price": "200.00"}
        ]
    else:
        positions_raw = get_current_positions(base_url, headers)
        
    for pos in positions_raw:
        symbol = pos["symbol"]
        qty = float(pos["qty"])
        market_val = float(pos["market_value"])
        price = float(pos["current_price"])
        current_positions[symbol] = {
            "qty": qty,
            "market_value": market_val,
            "price": price
        }
    
    print(f"Current holding assets: {list(current_positions.keys())}")
    if is_mocked:
        for sym, data in current_positions.items():
            print(f"  - {sym}: {data['qty']} shares @ ${data['price']:.2f} (Value: ${data['market_value']:,.2f})")
    else:
        for sym, data in current_positions.items():
            print(f"  - {sym}: {data['qty']} shares @ ${data['price']:.2f} (Value: ${data['market_value']:,.2f})")
            
    # 4. Determine trades
    sells = []  # List of (symbol, qty_to_sell, current_value, target_value)
    buys = []   # List of (symbol, target_value, current_value, buy_dollar_amount)
    
    # Check all target assets and current assets
    all_symbols = set(target_portfolio.keys()) | set(current_positions.keys())
    
    for symbol in all_symbols:
        target_w = target_portfolio.get(symbol, 0.0)
        target_val = target_w * equity
        
        current_pos = current_positions.get(symbol)
        current_val = current_pos["market_value"] if current_pos else 0.0
        current_qty = current_pos["qty"] if current_pos else 0.0
        current_price = current_pos["price"] if current_pos else None
        
        diff = target_val - current_val
        
        # We use a threshold to avoid tiny adjustments (e.g. less than $10 or 0.1% of portfolio)
        threshold = max(10.0, equity * 0.001)
        
        if abs(diff) < threshold:
            continue
            
        if diff < 0:
            # Need to sell/reduce
            if target_val == 0:
                # Liquidate fully
                sells.append((symbol, current_qty, current_val, 0.0, True))
            else:
                # Sell fraction
                if current_price is None or current_price <= 0:
                    if is_mocked:
                        current_price = 100.0
                    else:
                        current_price = get_latest_price(symbol, headers)
                if current_price:
                    qty_to_sell = abs(diff) / current_price
                    # Ensure we don't sell more than we own
                    qty_to_sell = min(qty_to_sell, current_qty)
                    sells.append((symbol, qty_to_sell, current_val, target_val, False))
                else:
                    print(f"Warning: Could not resolve price for {symbol}, skipping sell.")
        elif diff > 0:
            # Need to buy
            buys.append((symbol, diff, current_val, target_val))
            
    # 5. Execute Sells first
    print("\n--- CALCULATED SELL ORDERS ---")
    if not sells:
        print("No sell orders needed.")
    else:
        for symbol, qty, cur_val, tgt_val, is_liq in sells:
            action = "LIQUIDATE" if is_liq else "REDUCE"
            print(f"[{action}] {symbol} | Current: ${cur_val:,.2f} -> Target: ${tgt_val:,.2f} | Sell Qty: {qty:.4f}")
            if not dry_run:
                print(f"Placing SELL order for {symbol}...")
                resp = place_order(base_url, headers, symbol, "sell", qty=qty)
                if resp.status_code in [200, 201]:
                    print(f"Successfully placed sell order for {symbol}.")
                else:
                    print(f"Failed to place sell order for {symbol} (HTTP {resp.status_code}): {resp.text}")
                time.sleep(0.5)
                
    # 6. Execute Buys next
    print("\n--- CALCULATED BUY ORDERS ---")
    if not buys:
        print("No buy orders needed.")
    else:
        for symbol, amount, cur_val, tgt_val in buys:
            print(f"[BUY] {symbol} | Current: ${cur_val:,.2f} -> Target: ${tgt_val:,.2f} | Buy Value: ${amount:,.2f}")
            if not dry_run:
                print(f"Placing BUY order for {symbol}...")
                resp = place_order(base_url, headers, symbol, "buy", notional=amount)
                if resp.status_code in [200, 201]:
                    print(f"Successfully placed buy order for {symbol}.")
                else:
                    print(f"Failed to place buy order for {symbol} (HTTP {resp.status_code}): {resp.text}")
                time.sleep(0.5)
                
    if dry_run:
        print("\n*** DRY RUN MODE: No orders were actually placed. ***")
    else:
        print("\nRebalancing execution completed.")

if __name__ == "__main__":
    main()
