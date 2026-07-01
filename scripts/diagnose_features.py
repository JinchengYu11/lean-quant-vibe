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

def load_sample_constituent_data(data_dir, n_stocks=150):
    print(f"Loading daily data for {n_stocks} stocks...")
    zip_files = glob.glob(os.path.join(data_dir, "*.zip"))
    if not zip_files:
        print("Error: No daily constituent data found!")
        sys.exit(1)
        
    import random
    random.seed(42)
    selected_files = random.sample(zip_files, min(n_stocks, len(zip_files)))
    
    df_list = []
    for fpath in selected_files:
        ticker = os.path.basename(fpath).replace(".zip", "")
        try:
            with zipfile.ZipFile(fpath) as z:
                with z.open(f"{ticker}.csv") as csv_f:
                    df = pd.read_csv(csv_f, names=["date", "open", "high", "low", "close", "volume"])
                    df["symbol"] = ticker
                    df_list.append(df)
        except Exception:
            continue
            
    big_df = pd.concat(df_list, ignore_index=True)
    big_df["date"] = pd.to_datetime(big_df["date"])
    
    # Scale prices (QC format)
    for col in ["open", "high", "low", "close"]:
        big_df[col] = big_df[col] / 10000.0
        
    big_df = big_df.set_index(["symbol", "date"]).sort_index()
    return big_df

def main():
    data_dir = os.path.join(project_root, "data", "equity", "usa", "daily")
    big_df = load_sample_constituent_data(data_dir, n_stocks=150)
    
    print("Calculating factors (orthogonalize=False) to preserve raw correlations...")
    factors_df = calculate_factors(big_df, orthogonalize=False)
    factors_df = factors_df.astype(np.float32)
    print(f"Factors computed. Shape: {factors_df.shape}")
    
    # Compute labels
    print("Calculating labels...")
    close_matrix = big_df["close"].unstack(level=0).ffill()
    future_ret = close_matrix.shift(-21) / close_matrix - 1
    labels = future_ret.stack().swaplevel(0, 1).sort_index().astype(np.float32)
    labels.name = "target"
    
    # Align features and target
    train_data = pd.concat([factors_df, labels], axis=1).dropna()
    print(f"Aligned training data size: {train_data.shape}")
    
    X = train_data.drop(columns=["target"])
    y = train_data["target"]
    
    # Train LightGBM model with recommended hyperparams (CPU for diagnostic speed/compatibility)
    print("Training diagnostic LightGBM model...")
    model = lgb.LGBMRegressor(
        n_estimators=100,
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
    
    # 1. Feature Importances
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values(by="importance", ascending=False)
    
    # 2. Correlation Matrix analysis
    print("Computing feature correlation matrix...")
    corr_matrix = X.corr().abs()
    
    # Find highly correlated pairs
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    highly_corr_pairs = []
    for col in upper_tri.columns:
        corr_series = upper_tri[col]
        correlated = corr_series[corr_series > 0.85]
        for row, val in correlated.items():
            highly_corr_pairs.append((row, col, val))
            
    highly_corr_df = pd.DataFrame(highly_corr_pairs, columns=["feature_A", "feature_B", "correlation"]).sort_values(by="correlation", ascending=False)
    
    # Print results to stdout
    print("\n" + "="*50)
    print("TOP 20 MOST IMPORTANT FEATURES")
    print("="*50)
    print(importance_df.head(20).to_string(index=False))
    
    print("\n" + "="*50)
    print("BOTTOM 20 LEAST IMPORTANT FEATURES")
    print("="*50)
    print(importance_df.tail(20).to_string(index=False))
    
    print("\n" + "="*50)
    print(f"HIGHLY CORRELATED PAIRS (CORR > 0.85) - TOTAL {len(highly_corr_df)}")
    print("="*50)
    print(highly_corr_df.head(30).to_string(index=False))
    
    # Save diagnostic results
    reports_dir = os.path.join(project_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    diagnostic_out = {
        "importances": importance_df.to_dict(orient="records"),
        "highly_correlated": highly_corr_df.to_dict(orient="records")
    }
    
    out_path = os.path.join(reports_dir, "feature_diagnostic.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(diagnostic_out, f, indent=4, ensure_ascii=False)
    print(f"\nSaved complete diagnostic JSON report to {out_path}")

if __name__ == "__main__":
    main()
