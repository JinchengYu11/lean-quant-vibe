import pandas as pd
import numpy as np
import os

def load_fundamental_matrices(dates_index, symbols):
    """
    Load fundamental data from CSV files and align it to dates_index and symbols,
    ensuring no look-ahead bias by using NOTICE_DATE as disclosure time.
    """
    roe_matrix = pd.DataFrame(np.nan, index=dates_index, columns=symbols)
    net_profit_growth_matrix = pd.DataFrame(np.nan, index=dates_index, columns=symbols)
    revenue_growth_matrix = pd.DataFrame(np.nan, index=dates_index, columns=symbols)
    asset_turnover_matrix = pd.DataFrame(np.nan, index=dates_index, columns=symbols)
    
    fundamental_dir = r"d:\lean-quant-vibe\data\fundamental"
    cols = ['ROEJQ', 'PARENTNETPROFITTZ', 'TOTALOPERATEREVETZ', 'TOAZZL']
    
    for symbol in symbols:
        filepath = os.path.join(fundamental_dir, f"{symbol}.csv")
        if not os.path.exists(filepath):
            continue
        try:
            df = pd.read_csv(filepath)
            if df.empty:
                continue
            
            df['NOTICE_DATE'] = pd.to_datetime(df['NOTICE_DATE'])
            df['REPORT_DATE'] = pd.to_datetime(df['REPORT_DATE'])
            
            # Sort to handle duplicates and drop duplicates
            df = df.sort_values(['NOTICE_DATE', 'REPORT_DATE'])
            df = df.drop_duplicates(subset=['NOTICE_DATE'], keep='last')
            df = df.set_index('NOTICE_DATE')
            
            # Exact union and ffill alignment to prevent look-ahead bias
            union_idx = dates_index.union(df.index)
            
            present_cols = [c for c in cols if c in df.columns]
            df_union = df[present_cols].reindex(union_idx).ffill()
            df_final = df_union.loc[dates_index]
            
            for col in present_cols:
                if col == 'ROEJQ':
                    roe_matrix[symbol] = df_final[col]
                elif col == 'PARENTNETPROFITTZ':
                    net_profit_growth_matrix[symbol] = df_final[col]
                elif col == 'TOTALOPERATEREVETZ':
                    revenue_growth_matrix[symbol] = df_final[col]
                elif col == 'TOAZZL':
                    asset_turnover_matrix[symbol] = df_final[col]
        except Exception:
            pass
            
    # Cross-sectional median fillna for robust handling
    for mat in [roe_matrix, net_profit_growth_matrix, revenue_growth_matrix, asset_turnover_matrix]:
        medians = mat.median(axis=1)
        mat.update(mat.T.fillna(medians).T)
        mat.fillna(0.0, inplace=True)
        
    return roe_matrix, net_profit_growth_matrix, revenue_growth_matrix, asset_turnover_matrix

def load_consensus_matrices(dates_index, symbols):
    """
    Load consensus data from CSV files and calculate 4 rolling analyst expectations features:
    rating, coverage, consensus_growth, eps_revision.
    Ensures no look-ahead bias by using publishDate.
    """
    rating_matrix = pd.DataFrame(np.nan, index=dates_index, columns=symbols)
    coverage_matrix = pd.DataFrame(np.nan, index=dates_index, columns=symbols)
    growth_matrix = pd.DataFrame(np.nan, index=dates_index, columns=symbols)
    revision_matrix = pd.DataFrame(np.nan, index=dates_index, columns=symbols)
    
    consensus_dir = r"d:\lean-quant-vibe\data\consensus"
    cols = ['daily_count', 'rating_sum', 'eps_sum', 'eps_next_sum']
    
    for symbol in symbols:
        filepath = os.path.join(consensus_dir, f"{symbol}.csv")
        if not os.path.exists(filepath):
            continue
        try:
            df = pd.read_csv(filepath)
            if df.empty:
                continue
            
            df['publishDate'] = pd.to_datetime(df['publishDate'])
            df = df.dropna(subset=['publishDate'])
            if df.empty:
                continue
            
            # Convert types safely
            df['emRatingValue'] = pd.to_numeric(df['emRatingValue'], errors='coerce')
            df['predictThisYearEps'] = pd.to_numeric(df['predictThisYearEps'], errors='coerce')
            df['predictNextYearEps'] = pd.to_numeric(df['predictNextYearEps'], errors='coerce')
            
            # Map columns for daily sum aggregation
            df['rating_sum'] = df['emRatingValue']
            df['eps_sum'] = df['predictThisYearEps']
            df['eps_next_sum'] = df['predictNextYearEps']
            df['daily_count'] = 1
            
            df_daily = df.groupby('publishDate').agg({
                'rating_sum': 'sum',
                'eps_sum': 'sum',
                'eps_next_sum': 'sum',
                'daily_count': 'sum'
            })
            
            # Union index and align
            union_idx = dates_index.union(df_daily.index)
            df_union = df_daily.reindex(union_idx)
            
            for col in cols:
                df_union[col] = df_union[col].fillna(0.0)
                
            # Rolling window calculations
            roll_count_60 = df_union['daily_count'].rolling('60D').sum()
            roll_rating_sum_60 = df_union['rating_sum'].rolling('60D').sum()
            roll_eps_sum_60 = df_union['eps_sum'].rolling('60D').sum()
            roll_eps_next_sum_60 = df_union['eps_next_sum'].rolling('60D').sum()
            
            roll_count_30 = df_union['daily_count'].rolling('30D').sum()
            roll_eps_sum_30 = df_union['eps_sum'].rolling('30D').sum()
            
            # Compute rolling averages
            rating_avg_60 = roll_rating_sum_60 / (roll_count_60 + 1e-10)
            eps_avg_60 = roll_eps_sum_60 / (roll_count_60 + 1e-10)
            eps_next_avg_60 = roll_eps_next_sum_60 / (roll_count_60 + 1e-10)
            
            eps_avg_30 = roll_eps_sum_30 / (roll_count_30 + 1e-10)
            
            # Prior window values
            eps_sum_prior = roll_eps_sum_60 - roll_eps_sum_30
            count_prior = roll_count_60 - roll_count_30
            eps_avg_prior = eps_sum_prior / (count_prior + 1e-10)
            
            # Reindex to trading days and forward fill
            s_rating = rating_avg_60.loc[dates_index].ffill()
            s_coverage = roll_count_60.loc[dates_index].ffill()
            s_growth = ((eps_next_avg_60 - eps_avg_60) / (eps_avg_60.abs() + 1e-5)).loc[dates_index].ffill()
            s_revision = ((eps_avg_30 - eps_avg_prior) / (eps_avg_prior.abs() + 1e-5)).loc[dates_index].ffill()
            
            rating_matrix[symbol] = s_rating
            coverage_matrix[symbol] = s_coverage
            growth_matrix[symbol] = s_growth
            revision_matrix[symbol] = s_revision
            
        except Exception:
            pass
            
    # Cross-sectional median fillna
    for mat in [rating_matrix, coverage_matrix, growth_matrix, revision_matrix]:
        medians = mat.median(axis=1)
        mat.update(mat.T.fillna(medians).T)
        mat.fillna(0.0, inplace=True)
        
    return rating_matrix, coverage_matrix, growth_matrix, revision_matrix

def calculate_rsi(price_matrix, period=14):
    """
    向量化计算矩阵所有列的 RSI 指标
    """
    delta = price_matrix.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # 使用简单移动平均近似指数移动平均，在向量化计算中效率最高
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_factors(df: pd.DataFrame, orthogonalize=True) -> pd.DataFrame:
    """
    输入 QuantConnect 格式的多索引日线 DataFrame:
    Index: [Symbol, Time] (或 [Symbol, DateTime])
    Columns: [open, high, low, close, volume]
    
    输出相同多索引的 DataFrame，包含 97 个因子特征。
    """
    # 1. 将多索引 DataFrame 转换为 [Time, Symbol] 矩阵结构，提升矩阵运算性能
    open_matrix = df['open'].unstack(level=0)
    high_matrix = df['high'].unstack(level=0)
    low_matrix = df['low'].unstack(level=0)
    close_matrix = df['close'].unstack(level=0)
    volume_matrix = df['volume'].unstack(level=0)
    
    # 2. 基础日内/隔夜收益率矩阵计算
    daily_ret = close_matrix.pct_change()
    overnight_ret = open_matrix / close_matrix.shift(1) - 1
    intraday_ret = close_matrix / open_matrix - 1
    
    factor_dict = {}
    
    # (1) 经典时序动量 (ROC)
    for d in [1, 2, 5, 10, 20, 60, 120, 250]:
        factor_dict[f'roc_{d}'] = close_matrix.pct_change(d)
        
    # (2) 隔夜收益率滚动平均 (Overnight Momentum)
    for d in [1, 2, 5, 10, 20, 60, 120, 250]:
        factor_dict[f'overnight_ret_{d}'] = overnight_ret.rolling(d).mean()
        
    # (3) 日内收益率滚动平均 (Intraday Reversal/Momentum)
    for d in [1, 2, 5, 10, 20, 60, 120, 250]:
        factor_dict[f'intraday_ret_{d}'] = intraday_ret.rolling(d).mean()
        
    # (4) 收盘价偏离移动平均线比例 (Close / SMA)
    for d in [5, 10, 20, 60, 120, 250]:
        factor_dict[f'close_to_sma_{d}'] = close_matrix / close_matrix.rolling(d).mean() - 1
        
    # (5) 滚动标准差 (Volatility)
    for d in [5, 10, 20, 60, 120, 250]:
        factor_dict[f'vol_{d}'] = daily_ret.rolling(d).std()
        
    # (6) 收盘价偏离历史最高价比例 (Close / Max High)
    for d in [5, 10, 20, 60, 120, 250]:
        factor_dict[f'close_to_max_{d}'] = close_matrix / high_matrix.rolling(d).max() - 1
        
    # (7) 收盘价偏离历史最低价比例 (Close / Min Low)
    for d in [5, 10, 20, 60, 120, 250]:
        factor_dict[f'close_to_min_{d}'] = close_matrix / low_matrix.rolling(d).min() - 1
        
    # (8) 相对成交量占比 (Relative Volume)
    for d in [5, 10, 20, 60, 120, 250]:
        factor_dict[f'rel_volume_{d}'] = volume_matrix / volume_matrix.rolling(d).mean()
        
    # (9) 成交量滚动标准差 (Volume Volatility)
    for d in [5, 10, 20, 60]:
        factor_dict[f'volume_vol_{d}'] = volume_matrix.rolling(d).std()
        
    # (10) 价量滚动相关系数 (PV Correlation)
    for d in [10, 20, 60, 120]:
        factor_dict[f'pv_corr_{d}'] = close_matrix.rolling(d).corr(volume_matrix)
        
    # (11) 聪明钱代理指标 (Smart Money Q)
    # Q = Volume * Sign(Return) / Avg_Volume(20)
    daily_q = volume_matrix * np.sign(daily_ret) / (volume_matrix.rolling(20).mean() + 1e-10)
    for d in [5, 10, 20, 60]:
        factor_dict[f'smart_money_q_{d}'] = daily_q.rolling(d).mean()
        
    # (12) 下行半方差风险 (Downside Volatility)
    downside_ret = daily_ret.clip(upper=0)
    for d in [10, 20, 60, 120]:
        factor_dict[f'downside_vol_{d}'] = downside_ret.rolling(d).std()
        
    # (13) 波动的波动 (Volatility of Volatility, VoV)
    vol_20 = daily_ret.rolling(20).std()
    for d in [20, 60, 120]:
        factor_dict[f'vov_{d}'] = vol_20.rolling(d).std()
        
    # (14) 收益率滚动偏度 (Skewness)
    for d in [60, 120, 250]:
        factor_dict[f'skew_{d}'] = daily_ret.rolling(d).skew()
        
    # (15) 收益率滚动峰度 (Kurtosis)
    for d in [60, 120, 250]:
        factor_dict[f'kurt_{d}'] = daily_ret.rolling(d).kurt()
        
    # (16) Amihud 非流动性因子 (Amihud Illiquidity)
    am_daily = daily_ret.abs() / (volume_matrix + 1e-10)
    for d in [5, 10, 20, 60]:
        factor_dict[f'amihud_{d}'] = am_daily.rolling(d).mean()
        
    # (17) 影线特征 (Shadow Lines)
    upper_sh = (high_matrix - np.maximum(open_matrix, close_matrix)) / (close_matrix + 1e-10)
    lower_sh = (np.minimum(open_matrix, close_matrix) - low_matrix) / (close_matrix + 1e-10)
    for d in [1, 5, 20, 60]:
        factor_dict[f'upper_shadow_{d}'] = upper_sh.rolling(d).mean()
        factor_dict[f'lower_shadow_{d}'] = lower_sh.rolling(d).mean()
        
    # (18) 日内振幅特征 (Amplitude)
    amplitude = (high_matrix - low_matrix) / (close_matrix + 1e-10)
    for d in [1, 5, 20, 60]:
        factor_dict[f'amplitude_{d}'] = amplitude.rolling(d).mean()
        
    # (19) 震荡技术指标 (RSI)
    for d in [14, 28]:
        factor_dict[f'rsi_{d}'] = calculate_rsi(close_matrix, d)
        
    # (20) 经典趋势指标 (MACD)
    ema_12 = close_matrix.ewm(span=12, adjust=False).mean()
    ema_26 = close_matrix.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    factor_dict['macd_line'] = macd_line
    factor_dict['macd_signal'] = signal_line
    factor_dict['macd_hist'] = macd_line - signal_line
    
    # (21) 布林带偏离度 (Bollinger Band Deviation)
    mean_20 = close_matrix.rolling(20).mean()
    std_20 = close_matrix.rolling(20).std()
    bb_upper = mean_20 + 2 * std_20
    bb_lower = mean_20 - 2 * std_20
    factor_dict['bollinger_dev'] = (close_matrix - bb_lower) / (bb_upper - bb_lower + 1e-10)
    
    # (22) 真实波幅比率 (ATR Ratio)
    # TR = Max(High - Low, |High - Pre_Close|, |Low - Pre_Close|)
    tr = np.maximum(
        high_matrix - low_matrix,
        np.maximum(
            (high_matrix - close_matrix.shift(1)).abs(),
            (low_matrix - close_matrix.shift(1)).abs()
        )
    )
    for d in [14, 28]:
        factor_dict[f'atr_ratio_{d}'] = tr.rolling(d).mean() / (close_matrix + 1e-10)

    # (23) 财务基本面因子 (Fundamental Factors)
    roe_mat, net_profit_growth_mat, revenue_growth_mat, asset_turnover_mat = load_fundamental_matrices(close_matrix.index, close_matrix.columns)
    factor_dict['roe'] = roe_mat
    factor_dict['net_profit_growth'] = net_profit_growth_mat
    factor_dict['revenue_growth'] = revenue_growth_mat
    factor_dict['asset_turnover'] = asset_turnover_mat

    # (24) 分析师预期与预期修正因子 (Analyst Consensus & Revisions)
    rating_mat, coverage_mat, consensus_growth_mat, eps_revision_mat = load_consensus_matrices(close_matrix.index, close_matrix.columns)
    factor_dict['analyst_rating'] = rating_mat
    factor_dict['analyst_coverage'] = coverage_mat
    factor_dict['consensus_growth'] = consensus_growth_mat
    factor_dict['eps_revision'] = eps_revision_mat

    # (25) 动态加载自定义因子 (Custom Factors)
    try:
        from TmomAlpha.custom_factors import CUSTOM_FACTORS_REGISTRY
        for name, func in CUSTOM_FACTORS_REGISTRY.items():
            try:
                custom_val = func(open_matrix, high_matrix, low_matrix, close_matrix, volume_matrix)
                custom_val = custom_val.reindex(index=close_matrix.index, columns=close_matrix.columns)
                custom_val = custom_val.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
                factor_dict[name] = custom_val
            except Exception as e:
                print(f"Error calculating custom factor '{name}': {e}")
    except Exception as e:
        print(f"Error importing CUSTOM_FACTORS_REGISTRY: {e}")

    # 3. 将所有计算好的因子矩阵堆叠 (Stack) 并融合成 MultiIndex DataFrame

    stacked_series = []

    for name, mat in factor_dict.items():
        # stack() 将 [Time, Symbol] 转换成 [Time, Symbol] MultiIndex Series
        s = mat.stack()
        s.name = name
        stacked_series.append(s)
        
    # 合并成 DataFrame
    factors_df = pd.concat(stacked_series, axis=1)
    
    # 调整索引顺序，以匹配 QuantConnect 规范: [Symbol, Time]
    factors_df = factors_df.swaplevel(0, 1)
    factors_df = factors_df.sort_index()
    
    # 因子截面物理正交化 (Loewdin Symmetric Orthogonalization)
    def orthogonalize_factors(df_in):
        df_swapped = df_in.swaplevel(0, 1).sort_index()
        dates = df_swapped.index.unique(level=0)
        orth_dfs = []
        for d in dates:
            df_sec = df_swapped.loc[d]
            mat = df_sec.values
            
            mean = np.mean(mat, axis=0)
            std = np.std(mat, axis=0)
            std[std < 1e-10] = 1.0
            norm_mat = (mat - mean) / std
            
            cov = norm_mat.T @ norm_mat
            S, V = np.linalg.eigh(cov)
            S = np.clip(S, 1e-10, None)
            inv_sqrt_S = np.diag(1.0 / np.sqrt(S))
            trans_matrix = V @ inv_sqrt_S @ V.T
            orth_mat = norm_mat @ trans_matrix
            
            df_orth = pd.DataFrame(orth_mat, index=df_sec.index, columns=df_sec.columns)
            df_orth['date_col'] = d
            orth_dfs.append(df_orth)
            
        df_all = pd.concat(orth_dfs).reset_index()
        symbol_col = 'symbol'
        for c in ['symbol', 'Symbol', 'index', 'level_0', 'level_1']:
            if c in df_all.columns:
                symbol_col = c
                break
        df_all = df_all.rename(columns={symbol_col: 'symbol_col'})
        df_all = df_all.set_index(['symbol_col', 'date_col']).sort_index()
        df_all.index.names = df_in.index.names
        return df_all

    # 【修复关键】在进行 Loewdin 正交化前，必须先填充 NaN 和 inf，防止 NaN 时序或截面传播导致全矩阵变成 0.0
    factors_df = factors_df.replace([np.inf, -np.inf], np.nan)
    factors_df = factors_df.fillna(0.0)

    if orthogonalize:
        factors_df = orthogonalize_factors(factors_df)
    else:
        # 只做截面 z-score 归一化以对齐量纲，不进行正交化旋转
        def zscore_factors(df_in):
            df_swapped = df_in.swaplevel(0, 1).sort_index()
            dates = df_swapped.index.unique(level=0)
            z_dfs = []
            for d in dates:
                df_sec = df_swapped.loc[d]
                mat = df_sec.values
                mean = np.mean(mat, axis=0)
                std = np.std(mat, axis=0)
                std[std < 1e-10] = 1.0
                norm_mat = (mat - mean) / std
                df_z = pd.DataFrame(norm_mat, index=df_sec.index, columns=df_sec.columns)
                df_z['date_col'] = d
                z_dfs.append(df_z)
            df_all = pd.concat(z_dfs).reset_index()
            symbol_col = 'symbol'
            for c in ['symbol', 'Symbol', 'index', 'level_0', 'level_1']:
                if c in df_all.columns:
                    symbol_col = c
                    break
            df_all = df_all.rename(columns={symbol_col: 'symbol_col'})
            df_all = df_all.set_index(['symbol_col', 'date_col']).sort_index()
            df_all.index.names = df_in.index.names
            return df_all
        factors_df = zscore_factors(factors_df)
        
    return factors_df
