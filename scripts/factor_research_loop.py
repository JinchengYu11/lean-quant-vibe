#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import zipfile
import json
import re
import random
import time
import datetime
import subprocess
import shutil
import pandas as pd
import numpy as np
import requests

# Add project root to sys.path so we can import TmomAlpha modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# 1. Configuration (Dynamically resolved relative to project root)
DATA_DIR = os.path.join(project_root, "data", "equity", "usa", "daily")
CUSTOM_FACTORS_FILE = os.path.join(project_root, "TmomAlpha", "custom_factors.py")
ACCEPTED_FACTORS_JSON = os.path.join(project_root, "reports", "accepted_factors.json")
BACKTEST_SCRIPT = os.path.join(project_root, "scripts", "run_backtest.py")
BASELINE_SHARPE_JSON = os.path.join(project_root, "reports", "baseline_sharpe.json")
BACKTEST_METRICS_JSON = os.path.join(project_root, "reports", "backtest_metrics.json")

MAX_VALIDATION_STOCKS = 60  # Number of stocks to use for fast validation
IC_THRESHOLD = 0.008       # Minimum absolute Rank IC (0.8% is significant on A-shares)
IR_THRESHOLD = 0.05        # Minimum Information Ratio (5% is standard on validation sub-sets)
CORR_THRESHOLD = 0.70      # Maximum correlation with existing factors

def run_backtest_subprocess():
    """Runs run_backtest.py in a subprocess, printing live output, and returns the Sharpe ratio."""
    print("Launching full backtest simulation subprocess...")
    python_exe = os.path.join(project_root, ".venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = "python"
        
    try:
        # Run process and pipe output
        process = subprocess.Popen(
            [python_exe, BACKTEST_SCRIPT],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Print stdout live
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[Backtest] {output.strip()}")
                
        rc = process.poll()
        if rc != 0:
            stderr_out = process.stderr.read()
            print(f"Backtest process failed with exit code {rc}. Stderr:\n{stderr_out}", file=sys.stderr)
            
        # Read the metrics JSON
        if os.path.exists(BACKTEST_METRICS_JSON):
            with open(BACKTEST_METRICS_JSON, "r", encoding="utf-8") as f:
                metrics = json.load(f)
                return float(metrics.get("sharpe", 0.0))
    except Exception as e:
        print(f"Error running backtest subprocess: {e}", file=sys.stderr)
    return 0.0


def load_validation_data(n_stocks=MAX_VALIDATION_STOCKS):
    """
    Loads daily OHLCV data for a subset of stocks from ZIP files,
    aligns them into [Time, Symbol] price matrices.
    """
    print(f"Loading daily data for {n_stocks} validation stocks...")
    zip_files = glob.glob(os.path.join(DATA_DIR, "*.zip"))
    if not zip_files:
        print(f"Error: No data found in {DATA_DIR}. Please run download_csi1000.py first.", file=sys.stderr)
        sys.exit(1)
        
    # Sample a subset of zip files for speed
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
            
    if not df_list:
        print("Error: Failed to load any stock data.", file=sys.stderr)
        sys.exit(1)
        
    big_df = pd.concat(df_list, ignore_index=True)
    big_df["date"] = pd.to_datetime(big_df["date"])
    
    # Scale prices (LEAN formats prices multiplied by 10,000)
    for col in ["open", "high", "low", "close"]:
        big_df[col] = big_df[col] / 10000.0
        
    big_df = big_df.set_index(["symbol", "date"]).sort_index()
    
    # Unstack to [Time, Symbol] matrices
    open_matrix = big_df['open'].unstack(level=0)
    high_matrix = big_df['high'].unstack(level=0)
    low_matrix = big_df['low'].unstack(level=0)
    close_matrix = big_df['close'].unstack(level=0)
    volume_matrix = big_df['volume'].unstack(level=0)
    
    # Forward fill missing values to align matrices
    open_matrix = open_matrix.ffill().bfill()
    high_matrix = high_matrix.ffill().bfill()
    low_matrix = low_matrix.ffill().bfill()
    close_matrix = close_matrix.ffill().bfill()
    volume_matrix = volume_matrix.ffill().bfill()
    
    print(f"Data loaded successfully. Date range: {close_matrix.index.min().strftime('%Y-%m-%d')} to {close_matrix.index.max().strftime('%Y-%m-%d')}")
    return open_matrix, high_matrix, low_matrix, close_matrix, volume_matrix

def load_existing_factors():
    """Reads accepted_factors.json to check what factors we already have."""
    if os.path.exists(ACCEPTED_FACTORS_JSON):
        try:
            with open(ACCEPTED_FACTORS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def ask_gemini_for_factor(api_key, existing_factors):
    """Queries Gemini 2.5 Flash to generate a new A-share quantitative factor."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    existing_names = [f.get("name") for f in existing_factors]
    
    prompt = f"""你是一位资深的量化金融研究员。你需要设计一个新的A股选股因子（Alpha Factor）。
我们的目标是预测股票的**未来 21 天收益率（持仓期约为 1 个月）**。因此，因子设计应侧重于中短期（如5日、10日、20日）的累计价量特征、动量/反转趋势、价量背离、筹码分布代理指标、或中度周期的情绪指标，而不是极短期的单日盘口噪声。

我们已有的因子类别包括：
- 时序动量（ROC）、隔夜收益率平均、日内收益率平均
- 波动率（Volatility）、下行波动率、波动的波动（VoV）
- 聪明钱代理指标（Smart Money Q）、Amihud非流动性因子
- 收盘价偏离最高/最低价比例、相对成交量占比、价量相关性（PV Corr）
- 财务基本面（ROE、净利增长、营收增长等）与分析师预期修正（analyst rating, revision）

当前已定义的新因子名称列表：{existing_names}

请为我们设计一个不同于上述逻辑、基于日线量价数据且能预测未来 21 天中长线收益的**创新型因子**（例如基于多日累计价量背离强度、非对称波动动量、量价趋势协同、筹码锁定代理等）。

你的输出必须是合法的 JSON 格式（不要用 markdown ```json 标记包裹，直接输出 JSON 字符串），结构如下：
{{
  "factor_name": "以小写蛇形命名法命名的唯一因子名称，例如: volume_weighted_reversal",
  "description": "简要解释该因子的经济学逻辑和直觉（中文）。",
  "code": "一个完整的 Python 代码块。定义一个函数 `calculate_factor(open_matrix, high_matrix, low_matrix, close_matrix, volume_matrix)`。该函数接收 5 个 pandas DataFrame（索引为 date，列为 symbol）并返回一个相同形状的 DataFrame 包含因子值。必须使用 pandas 和 numpy 的向量化运算以保证效率。请妥善处理除零错误和 NaN（注意：在较新版本的 pandas 中，fillna(method='ffill') 已经被移除并会报错，请直接使用 DataFrame.ffill().bfill() 或 DataFrame.fillna(0.0) 进行空值处理）。"
}}
"""


    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "responseMimeType": "application/json"
        }
    }
    
    max_retries = 3
    for retry in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                candidates = result.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        return json.loads(parts[0].get("text", ""))
            elif response.status_code == 503:
                # API Overloaded retry with exponential backoff
                wait_sec = 3 + retry * 3
                print(f"Gemini API 503 Service Unavailable (spikes in demand). Retrying in {wait_sec} seconds...", file=sys.stderr)
                time.sleep(wait_sec)
                continue
            print(f"Error calling Gemini (HTTP {response.status_code}): {response.text}", file=sys.stderr)
            break
        except Exception as e:
            print(f"Error calling Gemini (attempt {retry+1}/{max_retries}): {e}", file=sys.stderr)
            time.sleep(2)
    return None

def evaluate_factor(factor_func, open_matrix, high_matrix, low_matrix, close_matrix, volume_matrix, future_returns):
    """
    Executes the factor function and calculates Rank IC, IC Std, and Information Ratio (IR).
    Also performs basic data sanity checks.
    """
    try:
        # Compute factor values
        factor_values = factor_func(open_matrix, high_matrix, low_matrix, close_matrix, volume_matrix)
        
        # 1. Sanity Checks (Deterministic Guards)
        if not isinstance(factor_values, pd.DataFrame):
            return None, "Output must be a pandas DataFrame"
        if factor_values.shape != close_matrix.shape:
            return None, f"Output shape {factor_values.shape} does not match input shape {close_matrix.shape}"
            
        # Clean infinite values and count NaNs
        factor_values = factor_values.replace([np.inf, -np.inf], np.nan)
        nan_ratio = factor_values.isna().sum().sum() / factor_values.size
        if nan_ratio > 0.30:
            return None, f"Too many NaNs in output: {nan_ratio:.2%}"
            
        # Fill remaining NaNs safely
        factor_values = factor_values.ffill().bfill().fillna(0.0)
        
        # Check if values are constant
        if factor_values.std().sum() < 1e-6:
            return None, "Factor output is constant or near-zero variance"
            
        # 2. Compute Rank IC (Spearman correlation with 21-day future returns)
        ic_series = factor_values.corrwith(future_returns, axis=1, method="spearman")
        ic_series = ic_series.dropna()
        
        if ic_series.empty:
            return None, "Empty Rank IC series (could be NaN alignment issue)"
            
        mean_ic = float(ic_series.mean())
        std_ic = float(ic_series.std())
        ir = mean_ic / std_ic if std_ic > 1e-8 else 0.0
        
        metrics = {
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "ir": ir,
            "nan_ratio": nan_ratio
        }
        return factor_values, metrics
    except Exception as e:
        return None, f"Runtime execution error: {str(e)}"

def check_collinearity(new_factor_values, open_matrix, high_matrix, low_matrix, close_matrix, volume_matrix):
    """
    Checks correlation with some base factors to prevent duplicate features,
    as well as with any existing custom factors.
    """
    # 1. Base factor: Simple 20-day return (momentum)
    base_mom = close_matrix.pct_change(20).ffill().bfill().fillna(0.0)
    
    # 2. Base factor: Simple 20-day volatility
    base_vol = close_matrix.pct_change().rolling(20).std().ffill().bfill().fillna(0.0)
    
    # Flatten and correlate
    new_flat = new_factor_values.values.flatten()
    
    max_corr = 0.0
    
    # Check against base factors
    for base_f in [base_mom, base_vol]:
        base_flat = base_f.values.flatten()
        mask = ~np.isnan(new_flat) & ~np.isnan(base_flat)
        if np.any(mask):
            corr = np.corrcoef(new_flat[mask], base_flat[mask])[0, 1]
            max_corr = max(max_corr, abs(corr))
            
    # Check against existing custom factors
    try:
        from TmomAlpha.custom_factors import CUSTOM_FACTORS_REGISTRY
        for name, func in CUSTOM_FACTORS_REGISTRY.items():
            try:
                existing_values = func(open_matrix, high_matrix, low_matrix, close_matrix, volume_matrix)
                existing_values = existing_values.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
                existing_flat = existing_values.values.flatten()
                mask = ~np.isnan(new_flat) & ~np.isnan(existing_flat)
                if np.any(mask):
                    corr = np.corrcoef(new_flat[mask], existing_flat[mask])[0, 1]
                    max_corr = max(max_corr, abs(corr))
            except Exception as e:
                print(f"Warning running existing custom factor '{name}' during collinearity check: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Warning loading custom factors registry during collinearity check: {e}", file=sys.stderr)
        
    return max_corr

def save_accepted_factor(factor_data, metrics, code):
    """Saves accepted factor code to custom_factors.py and logs metadata to JSON."""
    # 1. Update custom_factors.py
    # Format code nicely
    code_cleaned = code.strip()
    
    # Prefix every line of description with '#' to prevent SyntaxError
    desc_raw = factor_data.get('description', '')
    desc_lines = desc_raw.strip().split('\n')
    desc_comment = '\n'.join([f"# {line}" for line in desc_lines])
    
    with open(CUSTOM_FACTORS_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n\n# --- Accepted Factor: {factor_data['factor_name']} ---\n")
        f.write(f"{desc_comment}\n")
        f.write(f"# Metrics: Mean IC={metrics['mean_ic']:.4f}, IR={metrics['ir']:.4f}\n")
        f.write(code_cleaned)
        f.write(f"\nCUSTOM_FACTORS_REGISTRY['{factor_data['factor_name']}'] = {factor_data['factor_name']}\n")

        
    # 2. Update accepted_factors.json
    existing = load_existing_factors()
    new_entry = {
        "name": factor_data["factor_name"],
        "description": factor_data["description"],
        "mean_ic": metrics["mean_ic"],
        "ir": metrics["ir"],
        "added_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "code_snippet": code_cleaned
    }
    existing.append(new_entry)
    
    os.makedirs(os.path.dirname(ACCEPTED_FACTORS_JSON), exist_ok=True)
    with open(ACCEPTED_FACTORS_JSON, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4, ensure_ascii=False)
        
    print(f"SUCCESS: Factor '{factor_data['factor_name']}' accepted and written to files.")

def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)
        
    # 1. Load Data (Load once)
    open_matrix, high_matrix, low_matrix, close_matrix, volume_matrix = load_validation_data()
    future_returns = close_matrix.shift(-21) / close_matrix - 1
    
    # 2. Establish baseline Sharpe ratio if not already recorded
    baseline_sharpe = 0.0
    if os.path.exists(BASELINE_SHARPE_JSON):
        try:
            with open(BASELINE_SHARPE_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
                baseline_sharpe = float(data.get("baseline_sharpe", 0.0))
            print(f"Loaded baseline Sharpe ratio: {baseline_sharpe:.4f}")
        except Exception as e:
            print(f"Error loading baseline Sharpe: {e}. Re-calculating...", file=sys.stderr)
            
    if baseline_sharpe <= 0.0:
        print("No baseline Sharpe ratio found. Running initial backtest simulation to establish baseline...")
        baseline_sharpe = run_backtest_subprocess()
        if baseline_sharpe <= 0.0:
            print("Error: Failed to calculate baseline Sharpe ratio. Exiting.", file=sys.stderr)
            sys.exit(1)
        # Save baseline Sharpe
        os.makedirs(os.path.dirname(BASELINE_SHARPE_JSON), exist_ok=True)
        with open(BASELINE_SHARPE_JSON, "w", encoding="utf-8") as f:
            json.dump({"baseline_sharpe": baseline_sharpe}, f, indent=4)
        print(f"Baseline Sharpe ratio established and saved: {baseline_sharpe:.4f}")
        
    max_attempts = 15
    success = False
    
    for attempt in range(1, max_attempts + 1):

        print(f"\n=========================================")
        print(f"ATTEMPT {attempt} of {max_attempts}")
        print(f"=========================================")
        
        # 2. Load existing factors
        existing_factors = load_existing_factors()
        print(f"Currently have {len(existing_factors)} custom factors logged.")
        
        # 3. Request New Factor
        print("Calling Gemini 2.5 Flash to request new factor...")
        factor_data = ask_gemini_for_factor(api_key, existing_factors)
        if not factor_data:
            print("Failed to get factor design from Gemini. Skipping to next attempt.", file=sys.stderr)
            continue
            
        print(f"Proposed Name: {factor_data.get('factor_name')}")
        print(f"Proposed Description: {factor_data.get('description')}")
        
        # 4. Evaluate Factor Code
        code = factor_data.get("code", "")
        if not code:
            print("Error: No code provided. Skipping.", file=sys.stderr)
            continue
            
        func_name = factor_data.get("factor_name")
        # Ensure the function in the code block is named func_name
        code = re.sub(r'def\s+calculate_factor\b', f'def {func_name}', code)
        
        local_vars = {}
        try:
            exec(code, globals(), local_vars)
        except Exception as e:
            print(f"Compilation error: {e}. Skipping.", file=sys.stderr)
            continue
            
        factor_func = local_vars.get(func_name)
        if not factor_func:
            # Fallback check if it was defined under calculate_factor
            factor_func = local_vars.get("calculate_factor")
            if factor_func:
                print(f"Warning: function was compiled as calculate_factor, renaming in code block.")
                code = code.replace("def calculate_factor", f"def {func_name}")
                local_vars = {}
                try:
                    exec(code, globals(), local_vars)
                    factor_func = local_vars.get(func_name)
                except Exception as e:
                    print(f"Compilation error after renaming: {e}. Skipping.", file=sys.stderr)
                    continue
            
        if not factor_func:
            print(f"Error: Function '{func_name}' not found. Skipping.", file=sys.stderr)
            continue
            
        print("Running deterministic guards and calculating Rank IC / IR metrics...")
        factor_values, result = evaluate_factor(
            factor_func, open_matrix, high_matrix, low_matrix, close_matrix, volume_matrix, future_returns
        )
        
        if factor_values is None:
            print(f"REJECTED: Factor failed guards. Reason: {result}. Skipping.")
            continue
            
        print(f"Metrics: Mean Rank IC = {result['mean_ic']:.4f} | IC Std = {result['std_ic']:.4f} | IR = {result['ir']:.4f}")
        
        # 5. Collinearity Check
        print("Running collinearity check with base factors and existing custom factors...")
        max_corr = check_collinearity(factor_values, open_matrix, high_matrix, low_matrix, close_matrix, volume_matrix)
        print(f"Max correlation: {max_corr:.4f}")
        
        # 6. Final Decision Gate
        is_ic_ok = abs(result["mean_ic"]) >= IC_THRESHOLD
        is_ir_ok = abs(result["ir"]) >= IR_THRESHOLD
        is_corr_ok = max_corr <= CORR_THRESHOLD
        
        print("\n--- Decision Gate ---")
        print(f"1. Rank IC check (>= {IC_THRESHOLD}): {'PASS' if is_ic_ok else 'FAIL'} ({abs(result['mean_ic']):.4f})")
        print(f"2. IR check (>= {IR_THRESHOLD}): {'PASS' if is_ir_ok else 'FAIL'} ({abs(result['ir']):.4f})")
        print(f"3. Collinearity check (<= {CORR_THRESHOLD}): {'PASS' if is_corr_ok else 'FAIL'} ({max_corr:.4f})")
        print(f"----------------------")
        
        if is_ic_ok and is_ir_ok and is_corr_ok:
            print("\n--- Gate 1 Passed! Starting Gate 2 (Sharpe Ratio Validation) ---")
            
            # A. Create backup of custom_factors.py
            backup_path = CUSTOM_FACTORS_FILE + ".bak"
            shutil.copyfile(CUSTOM_FACTORS_FILE, backup_path)
            print(f"Backed up custom_factors.py to {backup_path}")
            
            # B. Temporarily save/append the factor to custom_factors.py
            save_accepted_factor(factor_data, result, code)
            
            # C. Run backtest simulation subprocess
            new_sharpe = run_backtest_subprocess()
            print(f"\nSharpe Gate comparison: New Sharpe = {new_sharpe:.4f} | Baseline Sharpe = {baseline_sharpe:.4f}")
            
            # D. Verification logic
            if new_sharpe > baseline_sharpe:
                print(f"SUCCESS: Sharpe Ratio improved! {new_sharpe:.4f} > {baseline_sharpe:.4f}. Factor accepted.")
                # Update baseline Sharpe
                baseline_sharpe = new_sharpe
                with open(BASELINE_SHARPE_JSON, "w", encoding="utf-8") as f:
                    json.dump({"baseline_sharpe": baseline_sharpe}, f, indent=4)
                # Clean up backup
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                success = True
                break
            else:
                print(f"REJECTED: Sharpe Ratio did not improve! {new_sharpe:.4f} <= {baseline_sharpe:.4f}. Rolling back...")
                # Restore backup
                shutil.copyfile(backup_path, CUSTOM_FACTORS_FILE)
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                # Rollback accepted_factors.json log entry
                try:
                    existing = load_existing_factors()
                    if existing and existing[-1]["name"] == factor_data["factor_name"]:
                        existing.pop()
                        with open(ACCEPTED_FACTORS_JSON, "w", encoding="utf-8") as f:
                            json.dump(existing, f, indent=4, ensure_ascii=False)
                        print("Rolled back accepted_factors.json log entry.")
                except Exception as e:
                    print(f"Error rolling back accepted_factors.json: {e}", file=sys.stderr)
                    
                print("Trying next candidate...")
                time.sleep(2)

            
    if not success:
        print(f"\nError: Exhausted all {max_attempts} attempts without finding an acceptable factor.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
