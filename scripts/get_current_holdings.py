import os
import sys
import glob
import zipfile
import json
import pandas as pd
import numpy as np
import lightgbm as lgb

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from TmomAlpha.factors import calculate_factors
from TmomAlpha.portfolio_optimizer import optimize_portfolio_weights

def load_all_data(data_dir):
    print("Loading data...")
    zip_files = glob.glob(os.path.join(data_dir, "*.zip"))
    all_dfs = []
    for idx, f in enumerate(zip_files):
        symbol = os.path.basename(f).replace(".zip", "")
        try:
            with zipfile.ZipFile(f) as z:
                with z.open(f"{symbol}.csv") as csv_f:
                    df = pd.read_csv(csv_f, names=["date", "open", "high", "low", "close", "volume"])
                    df["symbol"] = symbol
                    all_dfs.append(df)
        except Exception:
            pass
    big_df = pd.concat(all_dfs, ignore_index=True)
    big_df["date"] = pd.to_datetime(big_df["date"])
    for col in ["open", "high", "low", "close"]:
        big_df[col] = big_df[col] / 10000.0
    big_df = big_df.set_index(["symbol", "date"]).sort_index()
    return big_df

def main():
    data_dir = os.path.join(project_root, "data", "equity", "usa", "daily")
    big_df = load_all_data(data_dir)
    
    # Compute factors
    print("Calculating factors...")
    factors_df = calculate_factors(big_df, orthogonalize=False)
    factors_df = factors_df.astype(np.float32)
    
    # Compute labels
    close_matrix = big_df["close"].unstack(level=0).ffill()
    future_ret = close_matrix.shift(-21) / close_matrix - 1
    labels = future_ret.stack().swaplevel(0, 1).sort_index().astype(np.float32)
    labels.name = "target"
    
    # Get last rebalance date (first trading day of May 2026)
    all_dates = sorted(close_matrix.index)
    backtest_dates = [d for d in all_dates if d >= pd.Timestamp(2018, 1, 1) and d <= pd.Timestamp(2026, 5, 28)]
    
    # Rebalance dates are dates where month changes
    rebalance_dates = []
    last_month = 0
    for d in backtest_dates:
        if d.month != last_month:
            rebalance_dates.append(d)
            last_month = d.month
            
    last_rebal_date = rebalance_dates[-1]
    print(f"Latest rebalance date: {last_rebal_date.strftime('%Y-%m-%d')}")
    
    # Train GBDT model using data up to 2025-12-01 (same as 2026 training in backtest)
    train_end_date = pd.Timestamp(2025, 12, 31)
    train_label_end_date = train_end_date - pd.Timedelta(days=30)
    
    idx = pd.IndexSlice
    X_train = factors_df.loc[idx[:, :train_label_end_date], :]
    y_train = labels.loc[idx[:, :train_label_end_date]]
    train_data = pd.concat([X_train, y_train], axis=1).dropna()
    
    print(f"Training LightGBM model up to {train_label_end_date.strftime('%Y-%m-%d')}...")
    X = train_data.drop(columns=["target"])
    y = train_data["target"]
    
    model = lgb.LGBMRegressor(
        n_estimators=150,
        learning_rate=0.0421,
        num_leaves=210,
        max_depth=8,
        reg_alpha=205.7,
        reg_lambda=581.0,
        colsample_bytree=0.888,
        subsample=0.879,
        random_state=42,
        device='cpu',
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X, y)
    
    # Get metadata
    industry_map_path = os.path.join(project_root, "data", "stock_industry_map.json")
    shares_path = os.path.join(project_root, "data", "stock_shares_outstanding.json")
    
    with open(industry_map_path, "r", encoding="utf-8") as f:
        stock_industry_map = json.load(f)
    with open(shares_path, "r", encoding="utf-8") as f:
        stock_shares_outstanding = json.load(f)
        
    current_factors = factors_df.xs(last_rebal_date, level="date")
    active_tickers = [sym for sym in current_factors.index if sym in close_matrix.columns and pd.notna(close_matrix.at[last_rebal_date, sym])]
    
    X_predict = current_factors.loc[active_tickers]
    preds = model.predict(X_predict)
    pred_series = pd.Series(preds, index=active_tickers)
    
    # Run optimizer
    history_prices = close_matrix.loc[:last_rebal_date].tail(60)
    historical_returns = history_prices.pct_change().dropna(how='all')
    
    opt_weights = optimize_portfolio_weights(
        predictions=pred_series,
        current_factors=current_factors,
        close_prices=close_matrix.loc[last_rebal_date],
        float_shares_map=stock_shares_outstanding,
        stock_industry_map=stock_industry_map,
        historical_returns=historical_returns,
        max_weight=0.05,
        active_tickers=active_tickers,
        max_portfolio_size=30,
        style_bound=None,
        industry_bound=None
    )
    
    if opt_weights is None:
        print("Optimizer failed to find weights, falling back to top 30 equal weight.")
        sorted_preds = pred_series.sort_values(ascending=False)
        selected = sorted_preds.head(30).index.tolist()
        opt_weights = {t: 1.0 / 30 for t in selected}
        
    # Prepare results table
    results = []
    portfolio_dict = {}
    for sym, weight in sorted(opt_weights.items(), key=lambda x: x[1], reverse=True):
        industry = stock_industry_map.get(sym, "未知行业")
        pred_ret = pred_series.get(sym, 0.0)
        results.append({
            "code": sym,
            "weight": weight,
            "pred_ret": pred_ret,
            "industry": industry
        })
        portfolio_dict[sym] = weight
        
    # Write to reports/latest_holdings.json
    reports_dir = os.path.join(project_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    output_path = os.path.join(reports_dir, "latest_holdings.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    # Write to target_portfolio.json (which is used by rebalance_alpaca.py by default)
    target_portfolio_path = os.path.join(project_root, "target_portfolio.json")
    with open(target_portfolio_path, "w", encoding="utf-8") as f:
        json.dump(portfolio_dict, f, indent=4)
        
    print(f"\nSUCCESS: Calculated holdings list. Saved to {output_path} and {target_portfolio_path}")

if __name__ == "__main__":
    main()
