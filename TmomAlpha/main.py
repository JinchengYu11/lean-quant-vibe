from AlgorithmImports import *
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import os
import glob
import zipfile
import gc
from TmomAlpha.factors import calculate_factors

class CustomFeeModel(FeeModel):
    def GetOrderFee(self, parameters):
        # A-share average fee: 0.15% (commission + stamp duty)
        fee = parameters.Order.AbsoluteQuantity * parameters.Security.Price * 0.0015
        return OrderFee(CashAmount(fee, "USD"))

class TmomAlpha(QCAlgorithm):

    def Initialize(self):
        # 1. 设置日期范围（回测期：2018-01-01 到 2026-05-28）
        self.SetStartDate(2018, 1, 1)
        self.SetEndDate(2026, 5, 28)
        
        # 2. 设置初始资金 (RMB 10,000,000)
        self.SetCash(10000000)
        
        # 3. 策略参数
        self.portfolio_size = 30  # Top 30 选股
        self.min_expected_return = 0.0  # 绝对动量过滤
        
        # 4. 获取本地 A 股成分股并注册
        self.symbols_dict = {}
        data_dir = os.path.join(self.GetDataFolder(), "equity", "usa", "daily")
        
        zip_files = glob.glob(os.path.join(data_dir, "*.zip"))
        self.Log(f"Found {len(zip_files)} constituents in data directory.")
        
        tickers = []
        for f in zip_files:
            ticker = os.path.basename(f).replace(".zip", "")
            tickers.append(ticker)
            symbol = self.AddEquity(ticker, Resolution.Daily).Symbol
            self.symbols_dict[ticker] = symbol
            # 应用双边千分之一点五的费率模型
            self.Securities[symbol].SetFeeModel(CustomFeeModel())
            
        # 5. 读取历史数据并载入 Pandas
        self.Log("Loading daily data into Pandas DataFrame...")
        df_list = []
        for ticker in tickers:
            fpath = os.path.join(data_dir, f"{ticker}.zip")
            with zipfile.ZipFile(fpath) as z:
                with z.open(f"{ticker}.csv") as csv_f:
                    df = pd.read_csv(csv_f, names=["date", "open", "high", "low", "close", "volume"])
                    df["symbol"] = ticker
                    df_list.append(df)
                    
        big_df = pd.concat(df_list, ignore_index=True)
        big_df["date"] = pd.to_datetime(big_df["date"])
        
        # 价格缩放还原
        for col in ["open", "high", "low", "close"]:
            big_df[col] = big_df[col] / 10000.0
            
        big_df = big_df.set_index(["symbol", "date"]).sort_index()
        self.Log(f"Raw data shape: {big_df.shape}")
        
        # 6. 计算黄金因子特征库
        self.Log("Calculating 97 multi-scale factors...")
        factors_df = calculate_factors(big_df)
        
        # 内存优化：转为 float32 降低 50% 内存使用
        self.factors_df = factors_df.astype(np.float32)
        self.Log(f"Factors shape: {self.factors_df.shape}")
        
        # 7. 计算标签（未来 21 天收益率）
        self.Log("Calculating labels (21-day future return)...")
        self.close_matrix = big_df["close"].unstack(level=0)
        future_ret = self.close_matrix.shift(-21) / self.close_matrix - 1
        self.labels = future_ret.stack().swaplevel(0, 1).sort_index().astype(np.float32)
        self.labels.name = "target"
        
        # 清理无用大内存变量
        del big_df
        gc.collect()
        
        # 8. 缓存状态变量
        self.model_lgb = None
        self.model_xgb = None
        self.model_cb = None
        self.last_year = 0
        self.last_month = 0
        
        # 9. 初始训练（回测开始前，利用 2010 - 2017 年历史数据训练首期模型）
        initial_train_end = pd.Timestamp(2017, 12, 31)
        self.TrainModel(initial_train_end)
        
        # 初始化当前年份和月份，防止回测第一天重复触发
        self.last_year = 2018
        self.last_month = 1
        
        # 10. 加载行业和流通股本数据
        import json
        with open(r"d:\lean-quant-vibe\data\stock_industry_map.json", "r", encoding="utf-8") as f:
            self.stock_industry_map = json.load(f)
        with open(r"d:\lean-quant-vibe\data\stock_shares_outstanding.json", "r", encoding="utf-8") as f:
            self.stock_shares_outstanding = json.load(f)
            
        self.Log("Initialization completed successfully!")
        
    def TrainModel(self, end_date):
        """
        年度滚动重训模型函数。
        """
        end_date_ts = pd.Timestamp(end_date)
        self.Log(f"Retraining LightGBM model using data up to {end_date_ts.strftime('%Y-%m-%d')}...")
        
        idx = pd.IndexSlice
        # To strictly avoid look-ahead bias (data leakage) from overlapping 21-day forward labels,
        # we roll back the training end date by 21 trading days (approx. 30 calendar days).
        train_label_end_date = end_date_ts - pd.Timedelta(days=30)
        
        X_train = self.factors_df.loc[idx[:, :train_label_end_date], :]
        y_train = self.labels.loc[idx[:, :train_label_end_date]]
        
        # Filter completed rows and align
        train_data = pd.concat([X_train, y_train], axis=1).dropna()
        
        if len(train_data) < 1000:
            self.Log(f"Warning: Not enough training data ({len(train_data)} rows). Skipping retraining.")
            return
            
        X = train_data.drop(columns=["target"])
        y = train_data["target"]
        
        # 1. 训练 LightGBM
        self.model_lgb = lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            device='gpu',
            n_jobs=12,
            verbose=-1
        )
        try:
            self.model_lgb.fit(X, y)
        except Exception as e:
            self.Log(f"LightGBM GPU training failed: {e}. Fallback to CPU.")
            self.model_lgb = lgb.LGBMRegressor(
                n_estimators=150,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                device='cpu',
                n_jobs=12,
                verbose=-1
            )
            self.model_lgb.fit(X, y)
            
        # 2. 训练 XGBoost
        self.model_xgb = xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            device='cuda',
            n_jobs=12
        )
        try:
            self.model_xgb.fit(X, y)
        except Exception as e:
            self.Log(f"XGBoost GPU training failed: {e}. Fallback to CPU.")
            self.model_xgb = xgb.XGBRegressor(
                n_estimators=150,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                device='cpu',
                n_jobs=12
            )
            self.model_xgb.fit(X, y)
            
        # 3. 训练 CatBoost
        self.model_cb = cb.CatBoostRegressor(
            iterations=150,
            learning_rate=0.03,
            depth=5,
            subsample=0.8,
            bootstrap_type='Bernoulli',
            random_seed=42,
            task_type='GPU',
            thread_count=12,
            verbose=0
        )
        try:
            self.model_cb.fit(X, y)
        except Exception as e:
            self.Log(f"CatBoost GPU training failed: {e}. Fallback to CPU.")
            self.model_cb = cb.CatBoostRegressor(
                iterations=150,
                learning_rate=0.03,
                depth=5,
                subsample=0.8,
                bootstrap_type='Bernoulli',
                random_seed=42,
                task_type='CPU',
                thread_count=12,
                verbose=0
            )
            self.model_cb.fit(X, y)
            
        self.Log(f"Ensemble training complete. Total training samples: {len(train_data)}")
        
    def OnData(self, data: Slice):
        """
        每日数据更新入口。
        """
        current_year = self.Time.year
        current_month = self.Time.month
        
        # 1. 年度滚动重训（在新的一年首个交易日触发）
        if current_year != self.last_year:
            train_end_date = pd.Timestamp(year=current_year - 1, month=12, day=31)
            self.TrainModel(train_end_date)
            self.last_year = current_year
            
        # 2. 月度再平衡（在新的一月首个交易日触发）
        if current_month != self.last_month:
            self.last_month = current_month
            self.Rebalance()
            
    def Rebalance(self):
        """
        月度选股与仓位重置。
        """
        current_date_ts = pd.Timestamp(self.Time.date())
        self.Log(f"Executing monthly rebalance on {current_date_ts.strftime('%Y-%m-%d')}...")
        
        # 1. 提取当前日期截面的因子特征
        try:
            current_factors = self.factors_df.xs(current_date_ts, level="date")
        except KeyError:
            self.Log(f"No factor data available for date: {current_date_ts.strftime('%Y-%m-%d')}. Skipping rebalance.")
            return
            
        if current_factors.empty:
            self.Log("Current factors DataFrame is empty. Skipping rebalance.")
            return
            
        # 2. 模型预测
        symbols_in_factors = current_factors.index.tolist()
        tickers_to_predict = [t for t in symbols_in_factors if t in self.symbols_dict]
        
        if not tickers_to_predict:
            self.Log("No tradable tickers found in current factors. Skipping.")
            return
            
        X_predict = current_factors.loc[tickers_to_predict]
        preds_lgb = self.model_lgb.predict(X_predict)
        preds_xgb = self.model_xgb.predict(X_predict)
        preds_cb = self.model_cb.predict(X_predict)
        
        # 截面归一化 (Z-score) 并加权融合
        def zscore_sectional(v):
            mean = np.mean(v)
            std = np.std(v)
            if std < 1e-10:
                return np.zeros_like(v)
            return (v - mean) / std
            
        z_lgb = zscore_sectional(preds_lgb)
        z_xgb = zscore_sectional(preds_xgb)
        z_cb = zscore_sectional(preds_cb)
        
        blended_preds = 0.4 * z_lgb + 0.3 * z_xgb + 0.3 * z_cb
        
        # 建立映射 (由于标准化后的分值是以 0.0 为均值，所以过滤阈值 self.min_expected_return=0.0 正好代表过滤弱于截面均值的个股)
        pred_series = pd.Series(blended_preds, index=tickers_to_predict)
        
        # 3. 选股与过滤
        sorted_preds = pred_series.sort_values(ascending=False)
        
        # 过滤预测收益率非正的股票 (绝对动量过滤)
        positive_preds = sorted_preds[sorted_preds > self.min_expected_return]
        
        # 4. 凸优化选股和权重求解 (使用全市场 active_tickers 进行 Barra 中性化，并修剪至 portfolio_size)
        close_prices_dict = {}
        for t in tickers_to_predict:
            symbol = self.symbols_dict[t]
            price = float(self.Securities[symbol].Price)
            if price > 0:
                close_prices_dict[t] = price
        close_prices_series = pd.Series(close_prices_dict)
        tickers_with_price = [t for t in tickers_to_predict if t in close_prices_dict]
        
        # 计算过去 60 天的滚动收益率矩阵作为协方差收缩输入
        history_prices = self.close_matrix.loc[:current_date_ts].tail(60)
        historical_returns = history_prices.pct_change().dropna(how='all')
        
        from TmomAlpha.portfolio_optimizer import optimize_portfolio_weights
        
        opt_weights = None
        if tickers_with_price:
            opt_weights = optimize_portfolio_weights(
                predictions=pred_series.loc[tickers_with_price],
                current_factors=current_factors.loc[tickers_with_price],
                close_prices=close_prices_series,
                float_shares_map=self.stock_shares_outstanding,
                stock_industry_map=self.stock_industry_map,
                historical_returns=historical_returns,
                max_weight=0.05,
                active_tickers=tickers_with_price,
                max_portfolio_size=self.portfolio_size,
                style_bound=None,
                industry_bound=None
            )
            
        # 5. 计算目标权重 (已考虑优化器 fallback)
        target_weights = {}
        if opt_weights is not None:
            self.Log(f"Optimized portfolio selected {len(opt_weights)} stocks using Barra optimizer.")
            for t, weight in opt_weights.items():
                target_weights[self.symbols_dict[t]] = weight * 0.99
        else:
            selected_tickers = positive_preds.head(self.portfolio_size).index.tolist()
            self.Log(f"Optimizer fallback: Selected {len(selected_tickers)} equal-weight stocks.")
            if selected_tickers:
                target_weight = 1.0 / self.portfolio_size
                for t in selected_tickers:
                    target_weights[self.symbols_dict[t]] = target_weight * 0.99
                    
        # 获取所有当前持有或目标持有的股票
        all_symbols = set(target_weights.keys()) | {s for s in self.Portfolio.Keys if self.Portfolio[s].Invested}
        
        # 6. 执行交易并应用 A 股交易限制保护
        for symbol in all_symbols:
            target_w = target_weights.get(symbol, 0.0)
            current_w = self.Portfolio[symbol].Quantity * self.Portfolio[symbol].Price / self.Portfolio.TotalPortfolioValue if self.Portfolio[symbol].Invested else 0.0
            
            # 停牌检查: 在 Slice 中无数据或 volume 为 0
            is_suspended = (not self.CurrentSlice.ContainsKey(symbol)) or (self.CurrentSlice[symbol] is None) or (self.CurrentSlice[symbol].Volume == 0)
            
            is_limit_up = False
            is_limit_down = False
            if not is_suspended:
                bar = self.CurrentSlice[symbol]
                # 获取前一日收盘价
                t_list = [k for k, v in self.symbols_dict.items() if v == symbol]
                if t_list:
                    t = t_list[0]
                    try:
                        prev_close = self.close_matrix.loc[:current_date_ts].iloc[-2][t]
                    except (IndexError, KeyError):
                        prev_close = float(self.Securities[symbol].Price)
                else:
                    prev_close = float(self.Securities[symbol].Price)
                    
                if prev_close > 0:
                    is_limit_up = (bar.Close == bar.High) and (bar.Close / prev_close - 1.0 >= 0.098)
                    is_limit_down = (bar.Close == bar.Low) and (bar.Close / prev_close - 1.0 <= -0.098)
            
            final_w = target_w
            if is_suspended:
                final_w = current_w  # 停牌则保持当前仓位不变
            elif is_limit_up:
                if target_w > current_w:
                    final_w = current_w  # 涨停无法买入，限制增加仓位
            elif is_limit_down:
                if target_w < current_w:
                    final_w = current_w  # 跌停无法卖出，限制减少仓位
                    
            # 若偏离超过千分之一则进行调仓
            if abs(final_w - current_w) > 0.001:
                if final_w == 0.0:
                    self.Liquidate(symbol)
                else:
                    self.SetHoldings(symbol, final_w)
