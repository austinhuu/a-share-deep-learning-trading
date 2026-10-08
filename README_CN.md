# A 股深度学习量化交易研究系统

该仓库展示一个**基于日线行情的研究原型**：从 OHLCV 数据获取、27 个技术指标特征工程、LSTM / Transformer-LSTM / CNN1D / MLP 模型训练，到时间序列验证、滚动预测、风控回测和 VeighNa CSV 信号对接。

**请注意：** 这里的默认频率是**日线**，不是经过验证的分钟级 T+0 或实盘交易系统。VeighNa 端只监听已经生成的离线信号，**不会发送任何订单**。系统及所生成的指标仅供研究展示，不构成投资建议。

## 安装与运行

Python 3.10 或以上：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

无需 Token 的完整离线测试（**数据为合成数据，不代表真实市场表现**）：

```powershell
python run_stage_a.py --demo --mode walk_forward --model mlp --epochs 2 --test-bars 100
python scripts/plot_results.py
```

使用自己的授权日线行情 CSV（需要 `datetime,open,high,low,close,volume` 等列）：

```powershell
python run_stage_a.py --data-csv .\my_ohlcv.csv --mode walk_forward --model lstm --epochs 30
```

使用 Tushare（额外 `pip install tushare`）：

```powershell
$env:TUSHARE_TOKEN="你自己的私密Token"
python run_stage_a.py --source tushare --ticker 688981.SH --start 20220101 --end 20251231
```

开发阶段时序 CV + Optuna + 梯度与置换重要性：

```powershell
python run_stage_a.py --data-csv .\my_ohlcv.csv --mode full --model lstm --optuna-trials 10 --explain
```

按固定权重等权融合四个模型的独立 Walk-forward 预测：

```powershell
python run_stage_a.py --data-csv .\my_ohlcv.csv --mode ensemble
```

测试：`python -m pytest -q`。

## 研究验证边界

本版修复了标签尾部未知值、模型输入序列时间对齐、训练与验证/测试窗口的标签间隔、每个验证折按训练片段拟合标准化器，以及代码中的硬编码密钥与本地路径。旧版回测数值**不能直接沿用**；发布版没有声称某个具体夏普或收益率。

回测采用理想化收盘价成交假设，且不含真实 A 股 T+1 卖出限制、停牌、涨跌停成交约束、完整税费、冲击成本和交易排队。即使滚动预测是样本外，它的净值也不能被视为真实可成交的策略收益。

详见 [英文主说明](README.md)、[研究方法](docs/METHODOLOGY.md)、[更新记录](docs/RELEASE_NOTES.md) 与 [公开前核查清单](PUBLICATION_CHECKLIST.md)。

## 公开授权

本仓库不包含旧版原始市场 CSV、权重、缓存、明文 Token 或旧版实验指标。**公开前仍须确认实习单位代码/研究成果的发布权限，并撤销原项目包曾暴露的 Tushare Token。** 在权限未核实前，请把仓库设为 Private。
