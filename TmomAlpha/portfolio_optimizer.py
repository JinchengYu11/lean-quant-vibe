import os
import cvxpy as cp
import numpy as np
import pandas as pd

def optimize_portfolio_weights(predictions, current_factors, close_prices, float_shares_map, stock_industry_map, historical_returns, max_weight=0.05, active_tickers=None, max_portfolio_size=50, style_bound=0.1, industry_bound=0.02):
    """
    使用 cvxpy 对全市场/成分股集合权重进行凸优化求解，实现申万一级行业中性化与 Barra 8大风格中性化。
    优化目标：最大化预测超额收益 (Alpha)，同时最小化基于 Ledoit-Wolf 协方差收缩的主动风险。
    """
    if active_tickers is None:
        active_tickers = predictions.index.tolist()
        
    M = len(active_tickers)
    if M == 0:
        return {}
        
    # 提取当日的价格、因子值与股本数据，转为 numpy 数组并进行安全填充
    pred_values = np.nan_to_num(predictions.loc[active_tickers].values, nan=0.0)
    close_vals = np.nan_to_num(close_prices.loc[active_tickers].values, nan=0.0)
    
    # 提取风格因子值并安全填充
    mom_vals = np.nan_to_num(
        current_factors.loc[active_tickers, 'roc_250'].values if 'roc_250' in current_factors.columns else np.zeros(M),
        nan=0.0
    )
    vol_vals = np.nan_to_num(
        current_factors.loc[active_tickers, 'vol_60'].values if 'vol_60' in current_factors.columns else np.zeros(M),
        nan=0.0
    )
    liq_vals = np.nan_to_num(
        -current_factors.loc[active_tickers, 'amihud_60'].values if 'amihud_60' in current_factors.columns else np.zeros(M),
        nan=0.0
    )
    val_vals = np.nan_to_num(
        current_factors.loc[active_tickers, 'roe'].values if 'roe' in current_factors.columns else np.zeros(M),
        nan=0.0
    )
    gro_vals = np.nan_to_num(
        current_factors.loc[active_tickers, 'net_profit_growth'].values if 'net_profit_growth' in current_factors.columns else np.zeros(M),
        nan=0.0
    )
    lev_vals = np.nan_to_num(
        current_factors.loc[active_tickers, 'asset_turnover'].values if 'asset_turnover' in current_factors.columns else np.zeros(M),
        nan=0.0
    )
    
    # 计算所有匹配股票的流通股本中位数作为 fallback
    matched_float_shares = [v['float_shares'] for k, v in float_shares_map.items() if isinstance(v, dict) and 'float_shares' in v]
    median_float_shares = np.median(matched_float_shares) if matched_float_shares else 1.0e8
    
    float_shares = np.array([float_shares_map.get(t, {}).get('float_shares', median_float_shares) for t in active_tickers])
    float_mv = close_vals * float_shares
    
    # 1. Size (市值) 因子
    size_vals = np.nan_to_num(np.log(float_mv + 1e-10), nan=0.0)
    
    # 2. Beta (贝塔) 因子 (通过历史收益率相对于市场均值计算)
    if not historical_returns.empty and len(historical_returns) >= 10:
        market_ret = historical_returns.mean(axis=1)
        df_active = historical_returns.reindex(columns=active_tickers)
        cov_matrix_full = pd.concat([df_active, market_ret.rename('market')], axis=1).cov()
        cov_vector = cov_matrix_full.loc['market', active_tickers]
        market_var = market_ret.var()
        beta_vals = np.nan_to_num((cov_vector / (market_var + 1e-10)).values, nan=1.0)
    else:
        beta_vals = np.ones(M)
        
    # 对 8 大风格暴露进行 z-score 截面归一化
    def zscore(v):
        std = np.std(v)
        if std < 1e-6:
            return np.zeros_like(v)
        return (v - np.mean(v)) / std
        
    S_size = zscore(size_vals)
    S_beta = zscore(beta_vals)
    S_mom = zscore(mom_vals)
    S_vol = zscore(vol_vals)
    S_liq = zscore(liq_vals)
    S_val = zscore(val_vals)
    S_gro = zscore(gro_vals)
    S_lev = zscore(lev_vals)
    
    # 计算基准权重向量 (中证1000在该子集中的市值占比)
    float_mv_sum = np.sum(float_mv)
    if float_mv_sum <= 0:
        w_benchmark = np.ones(M) / M
    else:
        w_benchmark = float_mv / float_mv_sum
        
    # 提取行业信息并建立行业哑变量矩阵 H
    industries = []
    for t in active_tickers:
        ind_info = stock_industry_map.get(t, {})
        ind_name = ind_info.get('industry_name', '其他') if isinstance(ind_info, dict) else '其他'
        industries.append(ind_name)
        
    unique_industries = sorted(list(set(industries)))
    K = len(unique_industries)
    
    H = np.zeros((K, M))
    for i, ind in enumerate(industries):
        k = unique_industries.index(ind)
        H[k, i] = 1.0
        
    # 基准行业权重向量
    w_benchmark_industry = H @ w_benchmark
    
    # 3. Ledoit-Wolf 协方差收缩估计
    from sklearn.covariance import LedoitWolf
    if not historical_returns.empty and len(historical_returns) >= 10:
        ret_values = historical_returns.reindex(columns=active_tickers).fillna(0.0).values
        try:
            lw = LedoitWolf()
            cov_matrix = lw.fit(ret_values).covariance_
        except Exception as e:
            print(f"Ledoit-Wolf estimation failed: {e}. Fallback to Identity.")
            cov_matrix = np.eye(M)
    else:
        cov_matrix = np.eye(M)
        
    # 求解凸优化
    w = cp.Variable(M)
    
    # 目标函数：最大化预测收益，并使用 Ledoit-Wolf 协方差收缩矩阵对偏离施加二次惩罚
    objective = cp.Maximize(w @ pred_values - 0.02 * cp.quad_form(w - w_benchmark, cp.psd_wrap(cov_matrix)))
    
    # 约束条件列表
    constraints = [
        w >= 0,                     # 不允许卖空
        cp.sum(w) == 1.0,           # 满仓约束
        w <= max(max_weight, 1.1 / M) # 单股权重上限
    ]
    
    # 行业中性化约束
    if industry_bound is not None and industry_bound > 0:
        constraints.append(cp.norm_inf(H @ w - w_benchmark_industry) <= industry_bound)
    
    # 8大 Barra 风格中性化约束
    styles_list = [S_size, S_beta, S_mom, S_vol, S_liq, S_val, S_gro, S_lev]
    if style_bound is not None and style_bound > 0:
        for S_factor in styles_list:
            constraints.append(cp.abs(w @ S_factor - w_benchmark @ S_factor) <= style_bound)
        
    # 第一阶段求解
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception as e:
        print(f"Strict optimization solver error: {e}")
        pass
        
    # 第二阶段：如果无解，进行约束松弛
    if prob.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
        print("Warning: Strict optimization failed. Relaxing constraints...")
        constraints_relaxed = [
            w >= 0,
            cp.sum(w) == 1.0,
            w <= max(max_weight, 1.1 / M)
        ]
        if industry_bound is not None and industry_bound > 0:
            constraints_relaxed.append(cp.norm_inf(H @ w - w_benchmark_industry) <= max(0.04, industry_bound * 2.0))
        if style_bound is not None and style_bound > 0:
            for S_factor in styles_list:
                constraints_relaxed.append(cp.abs(w @ S_factor - w_benchmark @ S_factor) <= max(0.25, style_bound * 2.0))
            
        prob_relaxed = cp.Problem(objective, constraints_relaxed)
        try:
            prob_relaxed.solve(solver=cp.CLARABEL)
        except Exception as e:
            print(f"Relaxed optimization solver error: {e}")
            pass
            
        if prob_relaxed.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
            print("Warning: Relaxed optimization also failed. Reverting to fallback (benchmark weights).")
            return None
            
    # 提取非零权重并进行修剪
    weights_dict = {active_tickers[i]: float(w.value[i]) for i in range(M) if w.value[i] > 1e-5}
    sorted_weights = sorted(weights_dict.items(), key=lambda x: x[1], reverse=True)
    
    pruned_weights = []
    accum_weight = 0.0
    for ticker, wt in sorted_weights:
        if len(pruned_weights) < max_portfolio_size and wt >= 0.005:
            pruned_weights.append((ticker, wt))
            accum_weight += wt
        else:
            break
            
    if not pruned_weights or accum_weight < 0.3:
        print(f"Warning: Pruned portfolio weight too low ({accum_weight:.4f}). Reverting to fallback.")
        return None
        
    # 对保留股票重新归一化
    final_weights = {ticker: wt / accum_weight for ticker, wt in pruned_weights}
    return final_weights
