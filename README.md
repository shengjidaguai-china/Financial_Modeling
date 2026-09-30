<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!--                              README.md                                  -->
<!--              TrendCast Pro · A 股多周期预测与只读信号源                  -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->

<div align="center">

# 📈 TrendCast Pro

### A 股多周期方向预测 · 只读观测信号源

<p>
  <b>基于 LightGBM 的 5 / 10 / 20 日涨跌方向概率模型</b><br/>
  <sub>真实行情数据链路 · 无前视评估 · 门禁与风控建议 · 决策源契约 · 可离线复算</sub>
</p>

<sub>研究与工程实践用途 · 非商业许可 · <b>不构成投资建议</b> · 信号仅作观测参考</sub>

</div>

---

<div align="center">

<!-- ── 项目与状态 ── -->
<a href="./LICENSE"><img src="https://img.shields.io/badge/%E8%AE%B8%E5%8F%AF%E8%AF%81-%E7%A6%81%E6%AD%A2%E5%95%86%E7%94%A8-critical?style=flat-square&logo=creativecommons&logoColor=white" alt="License"></a>
<img src="https://img.shields.io/badge/%E7%89%88%E6%9C%AC-v2.0.0%20professional-1f6feb?style=flat-square" alt="Version">
<img src="https://img.shields.io/badge/%E7%8A%B6%E6%80%81-%E7%A0%94%E7%A9%B6%E9%98%B6%E6%AE%B5-d29922?style=flat-square" alt="Status">
<img src="https://img.shields.io/badge/%E4%BF%A1%E5%8F%B7-%E5%8F%AA%E8%AF%BB%E8%A7%82%E6%B5%8B%20%C2%B7%20%E4%B8%8D%E4%BD%9C%E4%BA%A4%E6%98%93%E4%BE%9D%E6%8D%AE-8b949e?style=flat-square" alt="Signal">

<br/>

<!-- ── 技术栈 ── -->
<img src="https://img.shields.io/badge/Python-%E2%89%A53.8-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/LightGBM-%E4%B8%BB%E5%8A%9B%E6%A8%A1%E5%9E%8B-025E8C?style=flat-square" alt="LightGBM">
<img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="sklearn">
<img src="https://img.shields.io/badge/pandas-%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86-150458?style=flat-square&logo=pandas&logoColor=white" alt="pandas">
<img src="https://img.shields.io/badge/NumPy-%E8%AE%A1%E7%AE%97-013243?style=flat-square&logo=numpy&logoColor=white" alt="numpy">
<img src="https://img.shields.io/badge/FastAPI-%3A8800-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
<img src="https://img.shields.io/badge/ONNX-%E5%AF%BC%E5%87%BA-005CED?style=flat-square&logo=onnx&logoColor=white" alt="ONNX">

<br/>

<!-- ── 数据源与能力 ── -->
<img src="https://img.shields.io/badge/%E6%95%B0%E6%8D%AE%E6%BA%90-Wind%EF%BC%88%E5%8F%AF%E9%80%89%EF%BC%89-9e6a03?style=flat-square" alt="Wind">
<img src="https://img.shields.io/badge/%E6%95%B0%E6%8D%AE%E6%BA%90-akshare-3fb950?style=flat-square" alt="akshare">
<img src="https://img.shields.io/badge/%E6%95%B0%E6%8D%AE%E6%BA%90-%E8%85%BE%E8%AE%AF%E8%B4%A2%E7%BB%8F-3fb950?style=flat-square" alt="Tencent">
<img src="https://img.shields.io/badge/%E5%85%9C%E5%BA%95-%E6%A8%A1%E6%8B%9F%E6%95%B0%E6%8D%AE%E4%B8%8D%E8%90%BD%E7%9B%98-8b949e?style=flat-square" alt="Simulation">
<img src="https://img.shields.io/badge/CLI-63%20%E5%91%BD%E4%BB%A4-8250df?style=flat-square&logo=gnubash&logoColor=white" alt="CLI">
<img src="https://img.shields.io/badge/%E6%B5%8B%E8%AF%95-78%20%E4%B8%AA%E6%B5%8B%E8%AF%95%E6%96%87%E4%BB%B6-3fb950?style=flat-square&logo=pytest&logoColor=white" alt="Tests">

<br/>

<!-- ── 可信度（如实标注） ── -->
<img src="https://img.shields.io/badge/AUC-0.52~0.57-d29922?style=flat-square" alt="AUC">
<img src="https://img.shields.io/badge/%E5%87%80%E8%B6%85%E9%A2%9D-%E6%97%A0%20edge-d29922?style=flat-square" alt="No edge">
<img src="https://img.shields.io/badge/%E9%98%B2%E7%9B%AE%E6%A0%87%E6%B3%84%E6%BC%8F-%E5%B7%B2%E4%BF%AE%E5%A4%8D-58a6ff?style=flat-square" alt="Leak fix">
<img src="https://img.shields.io/badge/%E6%9C%AC%E5%9C%B0%E5%A4%8D%E7%AE%97-%E5%86%BB%E7%BB%93%E5%BF%AB%E7%85%A7%E5%B0%B1%E7%BB%AA-3fb950?style=flat-square" alt="Reproducible">

<br/>

<!-- ── 美股扩展（akshare 免费 + fdnpy 期权） ── -->
<img src="https://img.shields.io/badge/%E7%BE%8E%E8%82%A1%E6%97%A5K-akshare-3fb950?style=flat-square" alt="US stock">
<img src="https://img.shields.io/badge/%E6%B5%B7%E5%A4%96%E6%9C%9F%E8%B4%A7-akshare-3fb950?style=flat-square" alt="US futures">
<img src="https://img.shields.io/badge/%E7%BE%8E%E8%82%A1%E6%9C%9F%E6%9D%83-fdnpy-9e6a03?style=flat-square" alt="US options">

<br/>

<!-- ── GitHub 动态徽章 ── -->
<a href="https://github.com/shengjidaguai-china/Financial_Modeling/stargazers"><img src="https://img.shields.io/github/stars/shengjidaguai-china/Financial_Modeling?style=flat-square&logo=github&label=Stars" alt="Stars"></a>
<a href="https://github.com/shengjidaguai-china/Financial_Modeling/network/members"><img src="https://img.shields.io/github/forks/shengjidaguai-china/Financial_Modeling?style=flat-square&logo=github&label=Forks" alt="Forks"></a>
<a href="https://github.com/shengjidaguai-china/Financial_Modeling/issues"><img src="https://img.shields.io/github/issues/shengjidaguai-china/Financial_Modeling?style=flat-square&label=Issues" alt="Issues"></a>
<img src="https://img.shields.io/github/last-commit/shengjidaguai-china/Financial_Modeling?style=flat-square&label=Last%20Commit" alt="Last Commit">
<img src="https://img.shields.io/github/repo-size/shengjidaguai-china/Financial_Modeling?style=flat-square&label=Size" alt="Repo Size">
<img src="https://img.shields.io/github/commit-activity/m/shengjidaguai-china/Financial_Modeling?style=flat-square&label=Commits/m" alt="Commit Activity">

</div>

---

## ⚡ 30 秒上手

> 完整跑通只需三步。**不需要任何 API Key** —— 默认走免费数据源；有 Wind 终端时才用得上 Key。

```bash
# 0 · 获取代码
git clone https://cnb.cool/yuppiez328/Financial_Modeling.git
cd Financial_Modeling

# 1 · 装依赖（核心四件套：akshare / pandas / numpy / pyarrow，无需任何 Key）
python -m venv .venv
.venv\Scripts\activate           # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install lightgbm             # 训练/评估必需（可选依赖，见下）

# 2 · 训练（采集 → 特征 → 训练 → 评估，自动完成）
python main.py train             # 首次约 2~6 分钟（采集占大头，需联网；本机实测 2 分 34 秒）

# 3 · 预测 / 起服务
python main.py predict 300308.SZ --horizon all    # 单只 · 全部周期
python main.py serve                              # REST API → http://localhost:8800
```

**没有模型也能先看看？** 不需要训练即可运行的自检命令：

```bash
python main.py --help            # 查看全部 63 个子命令
python main.py trials            # 评估试验登记（append-only）
python main.py release-check     # 发布态健康检查（读本地报告）
```

> 💡 **可选依赖按需安装**（不装则对应命令**明确报错**，不会静默跳过）：
> `lightgbm`（训练/评估/调参）· `optuna`（`tune` 超参搜索）· `hmmlearn`（`regime` 状态分层）·
> `mapie`（`conformal-interval` 保形区间）· `neuralforecast`（概率区间）。
> 详见 [`requirements.txt`](requirements.txt) 注释块。
>
> 💡 **Wind 数据源**：配置环境变量 `WIND_API_KEY` 即启用（机构级 P0 源）；未配置时自动跳过，
> 依次降级到 akshare / 腾讯财经，**无需任何 Key 即可跑通全链路**。

---

## 📊 模型评估（如实读，包括不好看的部分）

> **最新本地复跑**（2026-09-29 · 真实行情免费档（akshare / 腾讯财经） · 目标泄漏已修复）

**数据口径**：内建标的池 **38 只**（26 只 A股/ETF + 10 只商品期货主连 + 2 只外汇），
前复权日K，2020-01-01 起至最新交易日，按时序三段切分（train 70% / val 15% / test 15%），
测试样本 9041~9126 条。本次运行未配置 `WIND_API_KEY`，Wind 档自动跳过，实际走 **akshare / 腾讯财经免费档**。

| 模型周期 | 准确率 | AUC | 精确率 | 召回率 | 盈亏比(毛) | 评级 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 🟦 短期 5d | 49.19% | 0.5143 | 0.4918 | 0.9978 | 0.9681 | 接近随机 |
| 🟨 中期 10d | 47.70% | 0.5391 | 0.4770 | 1.0000 | 0.9121 | 低于随机 |
| 🟪 长期 20d | 46.99% | 0.5435 | 0.4681 | 0.9974 | 0.8863 | 低于随机 |

> ⚠️ **必须直说的两点**：
> 1. **召回率 ≈ 1.0 = 退化解**：三个模型几乎全部预测同一方向，而样本正类占比仅约 0.47~0.49，
>    因此准确率 ≈ 一直猜多数的得分，**准确率本身不携带信息**；判断区分度请看 AUC。
> 2. **本轮读数比 2026-09-09 那版更差**（当时 5d/10d/20d 准确率 52.7/53.2/51.3%、AUC 0.5688/0.5689/0.5420）。
>    差异来自**采集口径不同**（本次 38 只含期货/外汇、无 Wind Key 走免费档、数据截至 2026-09-29），
>    两组数字**不可直接比较**；两组都如实保留，不做择优展示。

**历史对照**（2026-09-09 · 26 只 A股/ETF · 腾讯财经前复权）

| 模型周期 | 准确率 | AUC | 胜率 | 盈亏比 | 夏普（近似） | 评级 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 🟦 短期 5d | 52.7% | 0.5688 | 52.7% | 1.11 | 0.39 | 弱（略优于随机） |
| 🟨 中期 10d | 53.2% | 0.5689 | 53.2% | 1.14 | 0.32 | 弱（略优于随机） |
| 🟪 长期 20d | 51.3% | 0.5420 | 51.3% | 1.05 | 0.09 | 接近随机 |

> ⚠️ 更早的评估（2026-06-27，10d/20d AUC 0.84）存在**目标泄漏**（`target` 列曾混入特征集）
> 与模拟数据口径问题，属虚高值，**不可比、不可引用**；根因已修复
> （`get_feature_columns` 强制排除 `target_*` 列）。

### 🔻 更硬的结论：这套信号**没有可用的决策增量**

九轮独立排查（2026-09-17 ~ 09-19，详见 [`cairn/`](cairn/) 知识层）用
「**相对全池等权的净超额**」作唯一记分板，结论一致：

- **绝对收益为正 ≠ 有 edge**：信号年化看着高于基准，但扣掉 `2×` 单边成本、做非重叠调仓后，
  **净超额三周期全部为负或不显著**（5d t −2.12 / 10d t −2.79 / 20d t −2.27）。
  随机抽同样数量标的，**5d 有 7.5% 的概率不亚于信号** ⇒ 选中的这些标的没有信息含量。
- **状态分层无解**：三分（牛/震荡/熊）与趋势/盘整二分，**无正向状态**（`conditional_edge_hint=False`）——
  「信号只藏在某种行情里」的假设**不被支持**。
- **加减特征无解**：八个特征子集**无一**把净超额推正（107 列冗余不是瓶颈）。
- **风险预警退路也堵死**：朴素 trailing-vol 基线对未来波动 IC 已有 **0.69~0.75**，
  模型输出的增量仅 **0.03~0.09**（低于效应量下限）⇒ 真要做波动/回撤预警，**用朴素基线即可**。
- **唯一方向为正的线索未达门槛**：`rank` 截面排序（每日 top-30% 做多）OOS 主动收益 +12.7pp、
  PSR 0.831，但 **DSR 0.099 仍 `insufficient_evidence`** ⇒ 维持 `report_only`，**不采信为正式证据**。

> ✅ **定位结论**：本项目的正确用法是「**可审计的信号研究工具 / 只读观测源**」，
> **不是**「高胜率赚钱机器」。所有信号出口均带 `position_role=observer`、`affects_gate=false`，
> **不产出仓位、不改门禁**。任何对外材料**不得**宣称高胜率或「即将解锁」。

**命中率审计（历史）** · 2026-09-09 记录：78 条预测到期 → 57 条经本地真实行情回溯 → 真实命中率 **56.1%**。
> ⚠️ 该次审计的明细（`logs/audit/predictions.jsonl`）现已不在库中（78 条全为 `verified=false`），
> **当前无法复现**，故仅作历史记录保留；需重新验证请运行 `python main.py audit`（需联网）。

---

## 🧩 它长什么样

<div align="center">

<table>
<tr>
<td width="50%"><img src="assets/readme/ui-data-pipeline.png" alt="真实行情数据链路" width="430"/><br/><sub><b>真实行情数据链路</b> — 前复权日K + 归一化净值对比</sub></td>
<td width="50%"><img src="assets/readme/ui-model-eval.png" alt="模型评估面板" width="430"/><br/><sub><b>模型评估面板</b> — 三周期六指标 + 波动率分位</sub></td>
</tr>
<tr>
<td width="50%"><img src="assets/readme/ui-console.png" alt="预测控制台" width="430"/><br/><sub><b>预测控制台</b> — 组合级摘要 + CLI 速查（<i>示例视图，非实时截图</i>）</sub></td>
<td width="50%"><img src="assets/readme/ui-architecture.png" alt="端到端架构" width="430"/><br/><sub><b>端到端架构</b> — 五层结构 + 只读信号源出口</sub></td>
</tr>
</table>

</div>

```text
数据采集 ──▶ 特征工程 ──▶ 模型训练 ──▶ 评估 ──▶ 导出 ──▶ 服务 ──▶ 报告
   │            │             │          │        │        │        │
 Wind        技术指标      LightGBM    双维指标   ONNX   FastAPI   日/周报
 akshare    量价波动       集成模型   泄漏检查   运行时  :8800     审计归档
 腾讯        防泄漏门控     TimesFM                            命中率回溯
 模拟兜底    时间特征                                                │
                                                             决策源契约出口
```

**数据源优先级链**（逐级降级，永不崩溃；未配置即跳过）

```text
Wind MCP (P0)  ──▶  akshare (P1)  ──▶  腾讯财经 (P1)  ──▶  模拟数据 (P6)
   需 WIND_API_KEY     免费 · 覆盖期货/外汇   免费 · 前复权      仅链路验证 · 不落盘
```

> akshare 是免费档中**唯一覆盖国内期货与外汇**的通道；未安装时静默跳过，链路不受影响。
> 模拟数据**绝不写入缓存**——不会污染真实历史（数据卫生硬约束）。

### 工程纪律（可核验）

| 纪律 | 实现 | 怎么验 |
|:---|:---|:---|
| **防目标泄漏** | `get_feature_columns` 强制排除 `target_*` 列 | `tests/test_label_leak_guard.py` |
| **特征契约对齐** | 训练端持久化 `feature_cols`，推理端按**名称+顺序**校验，缺列/错位 **fail-close 报错** | `tests/test_predictor_feature_alignment.py` |
| **模拟数据不落盘** | 兜底数据只用于链路验证，不写缓存 | `src/data/collector.py` |
| **推理缓存自动刷新** | 缓存过期经真实源刷新一次（fail-open）；**绝不采用模拟数据** | `src/inference/predictor.py` |
| **门禁 fail-close** | 门禁不可用 / 数据缺失 / 未达标 → 一律**不放行**，信号保持只读 | `src/trading/gate.py` |
| **无前视评估** | 时序切分 + CPCV 净化交叉验证（purge/embargo） | `python main.py overfit-audit` |

---

## 🖥️ 接口与交付形态

### REST API（FastAPI · 默认端口 8800）

| 方法 | 端点 | 说明 |
|:---:|:---|:---|
| `GET` | `/health` | 健康检查 |
| `GET` | `/api/v1/models` | 已加载模型清单 |
| `GET` | `/api/v1/predict/{symbol}` | 单只预测（`?horizon=`） |
| `POST` | `/api/v1/predict/batch` | 批量预测 |
| `GET` | `/api/v1/portfolio/summary` | 组合级摘要（多周期方向 + 概率） |
| `GET` | **`/api/v1/decision/feed`** | **决策源契约**：净看涨概率 / 综合分 / 校准概率 / 采纳建议 / 审计摘要 |
| `GET` | `/api/v1/signal/{symbol}` · `/api/v1/trade/{symbol}` | 交易信号 / 交易适配输出 |
| `GET` | `/api/v1/audit/report` · `/api/v1/audit/stats` | 审计报告 / 命中率统计 |

```bash
curl http://localhost:8800/health
curl "http://localhost:8800/api/v1/predict/300308.SZ?horizon=long_term"
curl "http://localhost:8800/api/v1/decision/feed?symbols=300308.SZ,510300.SH"

# 离线导出同一份契约（服务态与离线态逐字段一致）
python main.py decision-feed --stdout
```

### 决策源契约（`decision-feed/1`）

把「方向字符串 + 概率」的换算**收到生产方一侧**，下游不必自己重复实现（各写一套必然产生口径分歧）：

| 字段 | 含义 | 对下游的价值 |
|:---|:---|:---|
| `horizons.<h>.net_up_probability` | 每周期**净看涨概率** | **不必判断方向字符串**，不会被读成相反结论 |
| `horizons.<h>.calibrated_probability` / `uncertainty` | 概率校准层 | 是否标定一眼可见（缺失时如实 `calibration_applied=false`） |
| `aggregate.composite_score` / `composite_signed` | 多周期综合分（显式权重） | 权重口径统一；`missing_horizons` / `coverage` 如实披露 |
| `advisory.advisory_consumable` / `recommended_threshold` | 置信度采纳建议 | 门槛不再由消费方各拍一个数 |
| `audit` / `analytics` | 已回溯命中率 / 分档×已实现收益 | 是否采信有实证依据 |

**结构性纪律**：`position_role = observer`、`affects_gate = false`、`advisory_only = true`
—— **只读：不产出仓位、不改门禁**。缺失周期**不补 0.5**，非有限值一律 `None`。

> 📐 端到端的契约与边界说明见 [`cairn/decision-source-contract.md`](cairn/decision-source-contract.md)。

### TradingView 交付（`tv-export`）

把契约投影成 TradingView **可直接读入**的两件套：

| 入口 | 交付物 | 说明 |
|:---|:---|:---|
| 图片导入 | `signals/<symbol>.png` | 真 PNG，`tEXt` 块内嵌机器可读契约（全精度锚点） |
| Pine `request.seed` | `pine/trendcast/<symbol>.json` | 变量 × 时序表，`tv-pine/1` 格式 |

```bash
python main.py tv-export                             # 全池
python main.py tv-export --symbols 510300.SH         # 单标的
python main.py tv-export --no-anchors --card-limit 8 # 只出卡片
```

> **像素即契约**：卡片上的每个数值与同一份契约**逐字段相等**（守卫 `tests/test_tv_export.py`）。
> 不出口任何可交易字段（无仓位 / 无权重 / 无下单指令）。

---

## 🧰 常用命令（共 63 个，此处只列常用）

| 命令 | 说明 |
|:---|:---|
| `train` / `evaluate` | 完整训练流水线（采集 → 特征 → 训练 → 评估） |
| `predict <symbol>` / `batch <symbols...>` | 单只 / 批量预测（`--horizon {short_term,mid_term,long_term,all}`） |
| `serve` / `export` | 启动 API 服务 / 导出 ONNX |
| `daily-report` / `weekly-report` | 生成日 / 周度预测报告 |
| `audit` | 预测审计（真实行情回溯命中率） |
| `schedule` / `adaptive` | 自动重训练调度（常驻） / 自适应学习（性能监控 + 漂移检测） |
| `decision-feed` / `tv-export` | 决策源契约导出 / TradingView 交付（图片卡 + Pine） |
| `edge-check` | **基准相对决策增量**（净超额 + 扣成本 + 随机子集对照 + 状态分层） |
| `ablation` | 特征集 × 模型族联合消融 |
| `risk-signal` | 风险预测力检验（vs 朴素波动基线的增量） |
| `overfit-audit` | 过拟合审计（CPCV + DSR + PBO + 统一试验预算） |
| `regime` | 市场状态分层（HMM 牛/熊/震荡，`expanding` 无前视口径） |
| `calibration` / `conformal-interval` | 概率校准（isotonic / Platt） / 保形预测区间 |
| `portfolio-backtest` | 组合回测闭环（`--weights all` 三臂对照、`--signals rank` 截面排序） |
| `drift-monitor` / `feature-attribution` | 漂移监控（PSI / KS） / TreeSHAP 状态×特征归因 |
| `gate` / `gate-diagnose` / `risk-advice` | 策略门禁判定 / 阻塞诊断 / 风控建议（止损止盈，未放行则 fail-close） |
| `macro` / `monitor` / `ic` / `ic-trend` / `ic-pool` | 宏观数据 / 监控报表 / IC 与命中率门禁评估 |
| `trials` / `release-check` | 评估试验登记（append-only） / 发布态健康检查 |

> 运行 `python main.py --help` 查看全部命令与参数；`python main.py <command> --help` 查看单命令参数。

---

## 📁 项目结构

```text
.
├── configs/
│   ├── config.yaml              # 默认配置（simulation 单源，离线可跑）
│   └── config_pro.yaml          # 生产配置（wind→akshare→tencent→simulation，38 只标的）
├── data/
│   ├── raw/                     # 原始行情缓存 <symbol>.csv（模拟兜底不落盘）
│   │   └── *.frozen.20260917.csv    # 冻结日K快照 —— clone 后可离线复算
│   ├── processed/               # 特征工程后的数据集
│   └── news/                    # 新闻缓存
├── models/                      # 训练产物（pkl，含 feature_cols 契约）与 exported/（ONNX）
├── logs/                        # 运行日志与评估报告（evaluation_report.txt）
├── reports/                     # 报告输出（含 *.frozen.*.json 冻结读数）
├── src/
│   ├── data/                    # 采集 / 预处理 / 技术指标 / 情感分析 / 质量门控
│   ├── train/                   # LightGBM 训练器、模型定义、自适应学习
│   ├── eval/                    # 评估与研究方法层（双维评估器 + 30+ 专题模块）
│   ├── inference/               # 推理引擎（缓存刷新 + 特征按名对齐）
│   ├── api/                     # FastAPI 服务（含契约端点）
│   ├── export/                  # 契约导出 / ONNX
│   ├── report/                  # 日报 / 周报生成
│   ├── scheduler/               # 自动重训练调度
│   ├── audit/                   # 预测审计
│   ├── notification/            # 信号推送
│   ├── trading/                 # 交易适配层（信号 / 风控 / 订单 / 回测）
│   └── utils/                   # 通用工具
├── Kronos/                      # 第三方基础模型源码快照（见 docs/THIRD_PARTY.md）
├── scripts/                     # 占位模型生成等工具脚本
├── tests/                       # pytest 测试（78 个文件）
├── 00_kickoff/                  # 各阶段交付结论与决策材料（可追溯证据链）
├── cairn/                       # Project Cairn 知识层（ROADMAP / LOG / 专题结论）
├── main.py                      # CLI 入口
└── requirements.txt
```

---

## ⚙️ 配置要点

默认加载 `configs/config_pro.yaml`（存在时优先），可用 `--config <path>` 覆盖。

```yaml
data:
  source: ["wind", "akshare", "tencent", "simulation"]   # 按序回退；simulation 兜底不写缓存
  raw_dir: "data/raw"
  start_date: "2020-01-01"
  end_date: ""                    # 留空 = 采集至最新交易日
  split_ratio: {train: 0.7, val: 0.15, test: 0.15}       # 时序切分，不 shuffle
  markets:
    stock:   {enabled: true, symbols: [...]}   # 26 只 A股/ETF（12 只个股 + 14 只 ETF）
    futures: {enabled: true, symbols: [...]}   # 10 只商品期货主连（需 akshare）
    forex:   {enabled: true, symbols: [...]}   # 2 只外汇（需 akshare）

training:
  adaptive_learning:
    enabled: true
    drift_threshold: 0.05         # 准确率下降超阈值触发重训练
    auto_retrain: true
```

> 改标的池：直接编辑 `configs/config_pro.yaml` 的 `markets.*.symbols`，无需改代码。

### 🧪 可选模型后端

**TimesFM**（可插拔预测器）与 **Kronos**（第三方基础模型源码快照）均为可选的研究路径，
**默认关闭**、尚未接入主推理链路：

```powershell
# TimesFM（可选；未安装 PyTorch 时自动使用 models/timesfm_*.pkl 占位模型，仅供链路验证）
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install timesfm[torch]

# 启用后：
python main.py predict 600519.SH --horizon mid_term --model-type timesfm
python main.py predict 600519.SH --horizon mid_term --model-type ensemble   # LightGBM + TimesFM 融合
```

> - 占位模型：`models/timesfm_{short,mid,long}_term_*.pkl`（`{'model': DummyModel(), 'scaler': DummyScaler()}`），
>   确保无 PyTorch 环境下推理与测试仍可用；生成 / 修复脚本见 `scripts/create_timesfm_placeholders.py`。
> - 配置段：`model.timesfm.{enabled,context_days,verbose}`（见 `configs/config.yaml`）。
> - Windows 上若报 `WinError 126`，通常是 PyTorch 与 CUDA/CPU 版本不匹配，改用 CPU 版即可。
> - **Kronos/**：开源金融 K 线基础模型 [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)
>   的源码快照（commit `67b630e`），**非本项目原创**，仅作后续接入实验底座，尚未与 `src/` 主链路打通。

---

## 🧪 测试与复算

```bash
# 全量测试（78 个测试文件、1500+ 用例）
python -m pytest tests -q

# 离线复算：用冻结快照，无需联网、无需 Key
python main.py edge-check          # 基准相对净超额
python main.py ablation            # 特征集 × 模型族消融
python main.py risk-signal         # 风险预测力增量
```

> ⚠️ **测试环境提示**：本仓库根目录**不含** `pytest.ini` / `pyproject.toml`，
> 因此请显式指定 `tests` 目录（如上）。部分测试需要可选依赖（`optuna` / `hmmlearn` / `mapie`），
> 缺失时这些用例会**显式 skip 或 fail-import**，属预期行为、不是回归。
> 本机基线（2026-09-29，Python 3.11 + lightgbm 4.7.0）：**1542 passed / 10 failed / 8 skipped**，
> 10 条失败全部为**环境缺少可选依赖或依赖版本差异**所致。

> 🧊 **冻结数据可复算**：`data/raw/*.frozen.20260917.csv` 与 `reports/*.frozen.*.json`
> 已入库，clone 后**离线**即可复现关键读数，不依赖采集时刻与网络可达性。

---

## 🔬 研究进展（精简）

排期唯一事实来源：`schedule/plan.json`；历史试验以 append-only 登记在 `trials`。
各阶段结论与边界（含**否定结论**）沉淀在 [`cairn/`](cairn/) 与 [`00_kickoff/`](00_kickoff/)。

| 阶段 | 内容 | 关键结论（含否定） |
|:---|:---|:---|
| S1~S10 | 回测层 / baseline / 特征 / 风控 / 分池门禁 | 基础设施就绪（vectorbt 回测、按资产类别分池） |
| S11~S15 (G) | 评估量尺 / 标签重构 / qlib 因子 / 数据源升级 / 调参 | 换周期**不支持**；加横截面/宏观特征**无显著增量**；akshare 升 P1 覆盖期货外汇 |
| S16~S20 (H) | 保形区间 / CPCV 过拟合加固 / HMM 状态 / 概率校准 / LLM 辅助 | 校准有效（ECE 0.1042→0.0025）；状态分层解释力有限；LLM 辅助**评估不通过、整阶段取消** |
| S21~S25 (I) | 组合回测 / 三臂对照 / 漂移监控 / TreeSHAP / 组合级过拟合审计 | **PBO 0.011** 但 **DSR 0.08~0.24**、MinTRL ≫ 现有样本 ⇒ 样本长度不足以把 Sharpe>0 当真 |
| Issue #55（九轮） | 决策源契约 + 九轮有效性排查 + 全池基线冻结 | **无决策增量**（见上文）；两条静默缺陷已修；结论钉到同一全池切片并冻结入库 |
| Issue #66 (J) | Laya 只读第二决策源（接入前预注册 + 冻结快照回放） | 接入前**先冻结判据**（fail-close）；重型依赖与保留决策待人工签字 |

<details>
<summary><b>📌 点击展开：九轮排查如何一步步否掉各个假设</b></summary>

| 轮次 | 检验 | 结论 |
|:--:|:---|:---|
| ①②③ | 池共线性 / 三重障碍法标签 / 周期权重重排 | 缩池**无效**；标签口径**证伪**；现行权重与证据方向相反 |
| ④ | 波动分层与置信度语义 | 「高置信 ⇒ 负收益」只在**低波动**成立；置信度**不是** edge 信号 |
| ⑤ | 换记分板（相对全池等权净超额） | 信号**无净超额**，随机抽同数量标的即可反超 |
| ⑥ | 特征集 × 模型族联合消融 | 八个子集**无一转正**；模型族仅 10d「更不差」⇒ 不够格改配置 |
| ⑦⑧ | 状态分层（全池）+ 修复两条静默缺陷 | 各状态**净超额均为负**；顺带修掉 HMM 起步静默降级等真缺陷 |
| ⑧ | 风险预测力检验 | `no_risk_increment`：朴素波动基线已把模型能做的做完 |
| ⑨ | 全池基线读数冻结 | 结论钉到**同一全池切片**并冻结入库，clone 后离线可复算 |

</details>

---

## ⚠️ 已知限制

- **信号无正向 edge**（最硬结论）：净超额三周期为负或不显著，状态分层无正向状态，消融无子集转正
- **风险预警退路同样不成立**：朴素 trailing-vol 基线 IC 0.69~0.75，模型增量仅 0.03~0.09
- **模型存在退化倾向**：最新复跑三周期召回率 ≈ 1.0（近乎恒定方向输出）⇒ 准确率不携带信息
- **评估为历史回测口径**：未扣真实滑点与冲击成本，实盘前需纸面跟踪
- **样本长度不足**：MinTRL 2173~8382 天 ≫ 现有 1569 天；DSR 0.08~0.24
- **数据源依赖**：期货/外汇仅 akshare 档可覆盖；Wind 需终端与 Key（可选）
- **宏观与新闻情感为配置开关，默认关闭**；宏观缺失时**置空、不做前视填充**
- TimesFM / Kronos 路径尚未接入主推理链路（后者为第三方源码快照，仅实验底座）
- `tune` 的 optuna 搜索当前仅覆盖 LightGBM

---

## 🔒 许可证与免责声明

### 许可证

本项目采用 **禁止商业用途许可协议（Non-Commercial License）**，详见 [LICENSE](LICENSE)。

| | 条款 |
|:---:|:---|
| ✅ | **允许** — 学习、研究、教学、学术、非商业内部评估用途 |
| ❌ | **禁止** — 一切商业用途，包括商业产品 / 服务、金融机构对外产品、以营利为目的的量化交易或信号售卖 |
| ❌ | **禁止** — 转售、出租、分发牟利、再许可、去除版权标识 |
| 🔒 | **商用授权** — 须事先取得版权方（yuppiez328 / 安然）的书面同意 |

### 免责声明

本软件仅供学习、研究和非商业用途使用，按「现状」（AS IS）提供，**不构成任何投资建议**。
金融市场具有高度不确定性，任何模型的历史表现均不代表未来收益。
项目中出现的全部标的、数值与结论均为**研究与工程验证用途**。
使用者应自行承担一切投资风险与合规责任。

### 第三方组件

第三方依赖与源码快照的来源、许可证登记见 [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md)
与 [`Kronos/THIRD_PARTY_NOTICE.md`](Kronos/THIRD_PARTY_NOTICE.md)。

---

<div align="center">

<br/>

**Copyright © 2026 [yuppiez328（安然）](https://cnb.cool/yuppiez328) — 保留所有权利**

<sub>用数据说话 · 用量化决策 · 用纪律执行</sub>

<br/>

<img src="https://img.shields.io/badge/Made%20with-Python%20%26%20LightGBM-1f6feb?style=for-the-badge&logo=python&logoColor=white" alt="Made with Python">

</div>
