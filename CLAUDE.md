# CSI 1000 Quantitative Strategy · 项目宪法 (CLAUDE.md)

本文件是项目的「宪法」与开发指南。所有的模型训练、特征工程与组合优化决策必须与本文件对齐。

---

## 一、 项目简介

本项目是基于 QuantConnect Lean 架构的 **CSI 1000（中证 1000）指数增强量化策略**。它利用多模型 GBDT 机器学习方法预测个股超额收益 (Alpha)，并通过凸优化器 (CVXPY) 求解投资组合权重，实现超额收益最大化与主动风险控制。

---

## 二、 开发与运行指令

在开发或运行回测前，确保已激活虚拟环境并进入工作区：

### 2.1 环境与运行指令
- **运行本地策略诊断与回测**：
  ```powershell
  .venv\Scripts\python.exe C:\Users\Junof\.gemini\antigravity\brain\3279b692-0c6a-4f2a-b8f4-5d762cad7d76\scratch\generate_diagnostics.py
  ```
- **下载 CSI 1000 标的数据**：
  ```powershell
  .venv\Scripts\python.exe download_csi1000.py
  ```

### 2.2 数据目录结构
- **日线股票数据**：`data/equity/usa/daily/` (包含 CSI1000 标的 ZIP 包)
- **行业分类数据**：`data/stock_industry_map.json`
- **流通股本数据**：`data/stock_shares_outstanding.json`
- **财务基本面数据**：`data/fundamental/`
- **分析师预期数据**：`data/consensus/`

---

## 三、 核心策略技术决策

下列关键决策已经过实证对比与验证，**不要在没有人类同意的情况下推翻**：

### 3.1 因子特征库与正交化：使用原始因子
- **决策**：因子计算采用 `orthogonalize=False`（不进行 Loewdin 正交化旋转），仅做截面 Z-Score 归一化对齐量纲。
- **原因**：Loewdin 正交化会旋转特征空间，使单个因子（如动量 `'roc_250'`）混淆为所有特征的混合。这直接导致组合优化器提取的风格暴露变成杂讯，让风格中性化约束失效并吞噬 Alpha。

### 3.2 模型训练：微调超参数
- **决策**：在滚动训练中使用 Optuna 寻找的最佳超参数组合：
  - **LightGBM**：`learning_rate=0.0421`, `num_leaves=210`, `max_depth=8`, `reg_alpha=205.7`, `reg_lambda=581.0`, `colsample_bytree=0.888`, `subsample=0.879`。
  - **XGBoost**：`learning_rate=0.0421`, `max_depth=6`, `reg_alpha=10.0`, `reg_lambda=50.0`, `colsample_bytree=0.888`, `subsample=0.879`。
  - **CatBoost**：`learning_rate=0.0421`, `depth=6`, `l2_leaf_reg=30.0`。
- **原因**：默认的浅层树结构（如 `num_leaves=31`）会导致模型在 CSI 1000 复杂的微观结构特征上严重欠拟合，使年化收益降低约 3.6%。

### 3.3 交易保护：必须强制启用
- **决策**：在实盘与回测中必须强制执行 A 股交易保护（停牌期间不交易，涨停无法买入，跌停无法卖出）。
- **原因**：排除交易保护会导致“流动性幻觉”（回测能以收盘价瞬间买卖锁死股票），产生无法实盘落地的虚高收益。加入真实保护后，年化收益上限修正为约 **18.91%**。

### 3.4 组合优化：偏向释放与偏离惩罚
- **决策**：在优化器中将风格边界设为 `style_bound=None` 和 `industry_bound=None`，允许动量和市值风格自由偏离，同时利用 Ledoit-Wolf 协方差收缩矩阵对偏离施加二次惩罚。
- **原因**：策略主要 Alpha 源自对小市值和动量特征的重仓 tilt。若强行硬约束风格中性化，会彻底抹杀策略的超额收益（导致收益降至 7% 左右）。

---

## 四、 编码风格规范

- **Python 代码**：
  - 使用 Python 3.11+ 标准语法。
  - 变量和函数使用小写字母加下划线命名（snake_case），类名使用大驼峰（PascalCase）。
  - 对海量 Pandas DataFrame 运算进行内存优化（优先转换为 `float32`，及时调用 `gc.collect()`）。
- **注释规则**：
  - 核心计算、数学公式（如 Ledoit-Wolf、CVXPY 目标函数）必须写明设计原理与物理意义，而不只是解释代码本身。
  - 修改代码前，重读本文件，确认逻辑符合“项目宪法”定义的技术边界。
