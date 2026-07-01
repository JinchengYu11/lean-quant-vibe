# CSI 1000 Quantitative Strategy · 项目宪法 (CLAUDE.md)

本文件是项目的「宪法」与开发指南。任何后续接管本项目的 AI 实体或人类开发人员，必须严格遵循本宪法定义的开发规范与技术边界。

---

## 一、 项目简介

本项目是基于 QuantConnect Lean 架构的 **中证 1000 (CSI 1000) 指数增强量化策略**。
利用多模型 GBDT（LightGBM）机器学习方法在截面上预测个股未来 21 天的超额收益 (Alpha)，并通过凸优化器 (CVXPY) 对投资组合权重进行求解（限制个股最大权重 5%，以 Ledoit-Wolf 协方差收缩矩阵作为风险模型进行主动风险偏离惩罚），实现超额收益最大化与主动风险控制。

---

## 二、 开发与运行指令

所有脚本均已完成 Portable 改造，使用动态相对路径运行，支持开箱即用：

### 2.1 核心命令
- **运行完整滚动重训与回测**：
  ```powershell
  .venv\Scripts\python.exe scripts/run_backtest.py
  ```
  *该脚本会滚动重训 LightGBM，并在 A 股真实交易成本 (0.15%) 与交易保护 (停牌/涨跌停无法买卖) 约束下仿真运行，输出 `reports/detailed_quant_stats.json` 和 `reports/backtest_metrics.json`。*

- **刷新 QuantStats 性能图表及 HTML 报告**：
  ```powershell
  $env:PYTHONUTF8=1; .venv\Scripts\python.exe generate_quantstats_report.py
  ```
  *刷新 `reports/quantstats_report_30.html` 及 `figures/` 下的所有分析图表。*

- **特征分裂重要性与相关性排查**：
  ```powershell
  .venv\Scripts\python.exe scripts/diagnose_features.py
  ```
  *计算并生成 `reports/feature_diagnostic.json`。*

- **运行 AI 因子自动挖掘与评估闭环**：
  ```powershell
  $env:GOOGLE_API_KEY="您的GeminiKey"; $env:PYTHONUTF8=1; .venv\Scripts\python.exe scripts/factor_research_loop.py
  ```
  *结合 Gemini 2.5 Flash 因子构想，通过 Rank IC/IR 共线性校验（Gate 1）以及回测夏普改善校验（Gate 2）双重门禁自动追加/回退因子。*

- **获取当前最新截面目标持仓**：
  ```powershell
  .venv\Scripts\python.exe scripts/get_current_holdings.py
  ```
  *输出最新的 CVXPY 凸优化持仓，保存于 `reports/latest_holdings.json` 和 `target_portfolio.json`。*

- **实盘/模拟盘 Alpaca 调仓执行**：
  ```powershell
  $env:ALPACA_API_KEY="xxx"; $env:ALPACA_SECRET_KEY="xxx"; .venv\Scripts\python.exe rebalance_alpaca.py --dry-run
  ```

### 2.2 数据目录结构
- **日线股票数据**：`data/equity/usa/daily/` (包含 CSI 1000 标的 ZIP 压缩包)
- **行业分类数据**：`data/stock_industry_map.json`
- **流通股本数据**：`data/stock_shares_outstanding.json`
- **中证1000指数数据**：`data/csi1000_index.csv`

---

## 三、 核心策略技术决策

下列关键决策已经过数轮实证对比与验证，**不要在没有人类同意的情况下推翻**：

### 3.1 因子特征库与正交化：使用原始因子
- **决策**：因子计算采用 `orthogonalize=False`（不进行 Loewdin 正交化旋转），仅做截面 Z-Score 归一化对齐量纲。
- **原因**：Loewdin 正交化会旋转特征空间，使单个因子（如动量 `'roc_250'`）混淆为所有特征的混合。这直接导致组合优化器提取的风格暴露变成杂讯，让风格中性化约束失效并吞噬 Alpha。树模型（GBDT）天生对共线性极度鲁棒。

### 3.2 模型训练：微调超参数
- **决策**：在滚动训练中使用 Optuna 寻找的最佳超参数组合：
  - **LightGBM**：`learning_rate=0.0421`, `num_leaves=210`, `max_depth=8`, `reg_alpha=205.7`, `reg_lambda=581.0`, `colsample_bytree=0.888`, `subsample=0.879`。
- **原因**：默认的浅层树结构（如 `num_leaves=31`）会导致模型在 CSI 1000 复杂的微观结构特征上严重欠拟合，使年化收益降低约 3.6%。

### 3.3 交易保护 (Safeguards)：必须强制启用
- **决策**：在实盘与回测中必须强制执行 A 股交易保护（停牌期间不交易，涨停无法买入，跌停无法卖出）。
- **原因**：排除交易保护会导致“流动性幻觉”（回测能以收盘价瞬间买卖锁定股票），产生无法实盘落地的虚高收益。

### 3.4 组合优化：偏向释放与偏离惩罚
- **决策**：在优化器中将风格边界设为 `style_bound=None` 和 `industry_bound=None`，允许动量和市值风格自由偏离，同时利用 Ledoit-Wolf 协方差收缩矩阵对偏离施加二次风险惩罚。
- **原因**：策略主要 Alpha 源自对小市值和动量特征的重仓 tilt。若强行硬约束风格中性化，会彻底抹杀策略的超额收益。

---

## 四、 编码风格与开发规范

- **自定义因子管理**：
  - 新增因子必须写入 `TmomAlpha/custom_factors.py` 并且注册到 `CUSTOM_FACTORS_REGISTRY` 字典中。
  - 函数必须接受五个价格/成交量矩阵：`open_matrix, high_matrix, low_matrix, close_matrix, volume_matrix`，返回形状一致的 `pd.DataFrame` 因子值。
  - 必须对 NaN/Inf 进行前向/后向填充保护 (`df.ffill().bfill().fillna(0.0)`)。
- **Python 代码风格**：
  - 变量和函数使用小写加下划线（snake_case），类名使用 PascalCase。
  - 对海量 Pandas 运算进行内存优化（转换为 `float32`，及时调用 `gc.collect()`）。
- **持续维护**：
  - 修改代码前，请通读本文件，确认逻辑符合“项目宪法”定义的技术决策。
