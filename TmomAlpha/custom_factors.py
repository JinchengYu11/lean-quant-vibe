# -*- coding: utf-8 -*-
"""
This file is automatically managed by the AI Factor Research Loop.
All custom factors accepted by the system will be appended here.
"""

import pandas as pd
import numpy as np

# Custom factors registration dictionary
# Format: { 'factor_name': function_reference }
CUSTOM_FACTORS_REGISTRY = {}






# --- Accepted Factor: volume_weighted_range_momentum ---
# 捕捉股票在一段时间内（如10-20个交易日）由成交量加权的日内收盘价位置动量。当收盘价持续接近当日高点，且伴随高成交量时，表明市场存在持续的买入积累力量；反之，当收盘价持续接近当日低点，且伴随高成交量时，表明存在持续的卖出分配力量。该因子通过累加这种日内买卖压力，旨在识别机构或大户的筹码积累/派发行为，从而预测未来中短期（21天）的股价趋势。
# Metrics: Mean IC=-0.0148, IR=-0.0944
import pandas as pd
import numpy as np

def volume_weighted_range_momentum(open_matrix, high_matrix, low_matrix, close_matrix, volume_matrix):
    """
    计算 Volume-Weighted Range Momentum 因子。

    该因子旨在捕获股票在一段时间内由成交量加权的日内收盘价位置动量。
    当收盘价持续接近当日高点，且伴随高成交量时，表明市场存在持续的买入积累力量；
    反之，当收盘价持续接近当日低点，且伴随高成交量时，表明存在持续的卖出分配力量。
    该因子通过累加这种日内买卖压力，旨在识别机构或大户的筹码积累/派发行为，
    从而预测未来中短期（21天）的股价趋势。

    Args:
        open_matrix (pd.DataFrame): 开盘价矩阵。
        high_matrix (pd.DataFrame): 最高价矩阵。
        low_matrix (pd.DataFrame): 最低价矩阵。
        close_matrix (pd.DataFrame): 收盘价矩阵。
        volume_matrix (pd.DataFrame): 成交量矩阵。

    Returns:
        pd.DataFrame: 包含因子值的DataFrame。
    """
    # 确保所有输入DataFrame的形状和索引一致
    if not (open_matrix.shape == high_matrix.shape == low_matrix.shape == close_matrix.shape == volume_matrix.shape):
        raise ValueError("All input DataFrames must have the same shape.")
    if not (open_matrix.index.equals(high_matrix.index) and
            open_matrix.columns.equals(high_matrix.columns)):
        raise ValueError("All input DataFrames must have the same index and columns.")

    # 1. 计算每日的日内价格范围
    daily_range = high_matrix - low_matrix

    # 2. 计算收盘价在日内范围中的相对位置
    # 添加一个小的epsilon避免除以零。如果 daily_range 为 0，则 close - low 也为 0，
    # 此时 (close - low) / (daily_range + epsilon) 会得到一个接近 0 的值，表示没有明确的日内压力。
    epsilon = 1e-6 
    relative_close_position = (close_matrix - low_matrix) / (daily_range + epsilon)
    
    # 3. 计算每日的成交量加权日内收盘价位置动量 (BPS - Buying/Selling Pressure Score)
    # 缺失成交量视为0，以避免 NaN 传播。
    volume_matrix = volume_matrix.fillna(0.0)

    # daily_bps 值高表示强烈的买入积累，值低表示强烈的卖出分配
    daily_bps = relative_close_position * volume_matrix

    # 4. 计算滚动窗口内的累积 Volume-Weighted Range Momentum
    window = 10 # 默认使用10个交易日的窗口

    # 使用 rolling().sum() 计算累积值。min_periods=window 确保只有完整窗口的数据才生成因子值。
    factor_value = daily_bps.rolling(window=window, min_periods=window).sum()

    # 处理 NaN 值：滚动计算的结果，前 window-1 行会是 NaN。
    # 使用前向填充再后向填充，最后用 0 填充剩余的 NaN。
    factor_value = factor_value.ffill().bfill().fillna(0.0)

    return factor_value
CUSTOM_FACTORS_REGISTRY['volume_weighted_range_momentum'] = volume_weighted_range_momentum
