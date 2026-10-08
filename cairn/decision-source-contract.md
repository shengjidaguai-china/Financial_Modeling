# 决策源契约与有效性边界（只读决策源 → 下游消费者）

**当前真相**：16_ 对下游的交付物是 `contract_version = decision-feed/1` 的**只读决策源契约**
（`GET /api/v1/decision/feed`、`python main.py decision-feed`）。

## 一、为什么要有契约层（而不是让下游自己换算）

16_ 的定位是「只出方向与概率，决策权归下游」。但下游落地时每个消费方都自己重写一遍同样的换算，
于是**同一份预测在不同链路上被翻译成不同口径**：

| 环节 | 下游自算的口径 | 16_ 侧同义实现 |
|---|---|---|
| 多周期聚合权重 | tradingview `trendcast_signal_source._aggregate`：`short 0.2 / mid 0.5 / long 0.3` | `SignalEngine.DEFAULT_HORIZON_WEIGHTS`：`0.30 / 0.35 / 0.35` |
| 动作阈值 | `up_prob > 0.6 → BUY`、`< 0.4 → SELL`（就地硬编码） | 门禁 `min_hit_rate 0.52` / 置信度子集 thr ∈ [0.2,0.3] |
| 置信度口径 | `|up_prob − 0.5| × 2`（无命名） | `confidence_from_proba` / `uncertainty_from_probability` |
| 单周期方向+概率 | 自行判断 `direction == "看涨"`，否则取 `1 − p` | 无（此前未出口） |

**结论**：口径必须由生产方显式出口。契约层的核心字段 `net_up_probability` 就是
「每周期净看涨概率」——下游**不再需要判断 direction 字符串**，
`P(up)` 与「方向 + 该方向概率」不再可能被读成相反结论。

## 二、契约字段（增量，原始字段逐字段保留）

- `horizons.<h>.net_up_probability`：净看涨概率；`direction` 看跌时 = `1 − p`
- `horizons.<h>.calibrated_probability` / `calibration_applied` / `uncertainty`：S19 校准层
- `aggregate`：多周期加权综合分（`composite_score` ∈ [0,1]、`composite_signed` ∈ [-1,1]）、
  `coverage`、`missing_horizons`、`weights_used` —— **缺失周期不补 0.5**
- `advisory`：`advisory_consumable` / `recommended_threshold` / `calibrated_consumable`
- `audit`：已回溯命中率摘要（含 24h 窗口）
- `analytics`：置信度分档 × 已实现收益的联合分布 + 门槛扫描 + 保守判定
- 结构性纪律字段：`position_role = observer`、`affects_gate = false`、`advisory_only = true`

## 三、有效性边界（本轮最重要结论，如实入库）

用真实日K + 本地真实训练的 LightGBM，按每 5 交易日一个锚点回填已实现收益
（15584 锚点，26 标的，3 周期；`reports/decision_feed/analytics_backfill.json`）：

| 置信度档 | 覆盖率 | 命中率 | 平均已实现收益 |
|---|---|---|---|
| [0.0,0.1) | 22.7% | 54.0% | **+1.92%** |
| [0.1,0.2) | 21.0% | 58.9% | **+1.69%** |
| [0.2,0.3) | 19.1% | 61.7% | **+1.08%** |
| [0.3,0.5) | 25.4% | 71.1% | **+0.51%** |
| [0.5,0.7) | 10.3% | 82.9% | **−0.49%** |
| [0.7,1.0] | 1.4% | **98.2%** | **−0.78%** |

**置信度越高 → 方向命中率越高，但平均已实现收益越低（单调反向）。**
杠杆口径下同样反向（`mean_return / mean_abs_return`：低档 +0.355、高档 −0.080）。
三周期判定均为 `ineffective`。

机制解释：模型在趋势加速段最自信，而该段恰是**短期已过热、后续均值回复**的位置
（极端档 mean_abs_return 0.097 vs 低档 0.054，波动近乎翻倍而方向仍对 —— 典型的高位高波动）。

**对下游的含义**：
- 高置信 ≠ 可采信；**「命中率高」不能单独作为采信依据**，必须与收益联合判定
  （判定函数强制两条腿都过，见 `decision_analytics.verdict`）；
- 建议门槛因此**不能**按「命中率最高」来挑；本契约的 `recommended_threshold=0.2`
  是「已配置现行值」（S15/G5 口径，T15.3=defer），不是本层择优结果；
- 结论为 `ineffective` 时，下游应继续把信号当**只读观测 / 反向参考 / 风险预警**，
  不得据此放大仓位。

**边界声明**：锚点间有重叠（每 5 交易日取值、5/10/20 日视界），水平读数不可当独立样本；
但方向性结论在重叠下依然成立且更强。该结论**不改变** 16_ 任何门禁判定
（`affects_gate=false`），是否据此调整门槛属人工检查点。

## 四、踩坑与解法（contains）

- **contains 推理特征位置截断会静默错位**：`_align_features` 在「特征列数相同但列序/列集合不同」时
  **不报警不报错**，把 A 列值喂给期望 B 列的模型，产出看似正常的错概率。现实触发路径：
  `evaluate_models.build_supervised` 的列集合（含 `_symbol` / `_fwd_ret`）与推理期
  `FeatureEngineer.get_feature_columns` 不逐字相同。
  解法：`_align_by_feature_names` —— 模型有 `feature_name_` 时按名取列并按模型顺序排列，
  缺列 0 填充 + WARNING 留痕；无特征名回落位置逻辑。守卫：`tests/test_predictor_feature_alignment.py`。
- **contains 共享记账接口字段口径不同会静默写坏记录**：下游按逐字段 dict 送预测，
  与 `record_prediction(prediction_dict)` 口径不同 → 被写成 `symbol=None` 的坏记录，
  不报错也无法分辨来源。解法：`PredictionAudit.record_prediction_v2` 关键字段缺失 fail-loud，
  来源显式入库。守卫：`tests/test_audit_record_v2.py`。
- **contains 0.5 不能既当"中性读数"又当"数据坏了"**：契约层所有换算在缺失/非有限值上一律 `None`，
  只有真实算出来的 0.5 才是中性读数（`net_up_probability` / `aggregate` 同此纪律）。

## 五、下游接入方式

```bash
# 服务态（推荐：与本地 :8800 链路同源）
curl "http://127.0.0.1:8800/api/v1/decision/feed?symbols=300308.SZ,510300.SH"

# 离线管道态（tradingview scripts/ 逐行读标的清单的用法）——离线/服务态**逐字段一致**
python main.py decision-feed --symbols-file ~/positions.txt --stdout
```

原 `/api/v1/portfolio/summary` 亦已自动追加决策字段（向后兼容，旧消费方零改动）。

**下游系统（独立仓）消费端状态（2026-10-03 升级交付）**：
- 客户端 `get_decision_feed()` 优先打 `/api/v1/decision/feed`（校验 `contract_version`），
  回退 `portfolio/summary`（亦为契约超集）→ 一期 batch 组合；全程 fail-open。
- 批量端点超时分层 `batch_timeout=900s`：**实测 26 标的全池 feed ≈ 7 分钟**
  （数据管线加重后单周期推理 ~5.4s；一期 10s 超时假设已被打破，属如实入库的回归发现）。
  生产端配套加了 `PredictionEngine` 预测缓存（键含数据最后日期+行数，新数据自动失效、
  error 不缓存；`tests/test_predictor_prediction_cache.py` 7 例）——同日重复调用毫秒级。
- 快照（`signals_YYYY-MM-DD.json`）增量携带契约字段（一期键名保留）；信号卡与简报呈现
  `advisory` 采纳建议与 `analytics.verdict` 有效性边界——**采纳建议 0 达标或 verdict
  非 effective 时明示「信号仅作只读观测/风险预警，不作为仓位依据」**（把第三节纪律
  变成下游展示层的内建行为，不再依赖下游自行记得）。交付记录见对接设计方案 §10。

## 六、交付形态投影（2026-09-16 追加）

本节结论**不改**上面任何内容，只补「下游怎么把它读进去」。TradingView 侧两条近原生入口：

| 入口 | 本项目交付物 | 生成 |
|---|---|---|
| 客户端图片导入（读一张静态 PNG） | `signals/<symbol>.png`（`tEXt` 内嵌 `signal_contract` + `anchors_json`） | `python main.py tv-export` |
| Pine `request.seed`（读变量×时序表） | `pine/trendcast/<symbol>.json`（`tv-pine/1`） | 同上 |

两个硬约束（不是风格选择，是消费方限制）：TradingView **不做二次渲染** → 图片必须是真 PNG；
Pine 只能读表结构 → JSON 必须带**列字典**（`columns[].id` 与列名逐字一致）。

**新增读数（38 标的池，3420 锚点，真实日K + 真实训练 LightGBM）**：
锚点命中 52.9% / 平均已实现收益 +0.32%；三周期置信度几乎全部贴地（`|composite-0.5|` ≤0.02），
`advisory_consumable` **0/38**，AUC ≈ 0.50~0.54。
即：第五节「高置信 ≠ 可采信」这条结论**依然成立，而且当前连"高置信"样本都还产不出来** ——
门槛如实挡住全部信号，不是缺陷。

**新增踩坑（与第五节同一条纪律的又一次翻车）**：
`_align_by_feature_names` 的调用方把**值矩阵**当列名传入 → 全列判缺失 → 整体 0 填充 →
产出与标的/日期无关的**常数概率**（全池锚点置信度恒 0.0303756，**不报错**）。
修法：全列缺失时**放弃按名对齐**（回落位置对齐）而不是 0 填充；训练侧把 `feature_cols`
写进产物并同步为模型原生特征名（`feature_name_` 只读 + booster 缓存两条实测约束）。
详见 `cairn/tradingview-handoff.md`。
## 七、Issue #55 正式关闭（2026-10-08 用户确认）

**关闭依据**：9 轮排查一致结论 + 用户 2026-10-08 确认。

| 线索 | 轮次 | 结论 |
|---|---|---|
| 池共线性 | ① | 缩池无效 |
| 三重障碍法标签 | ② | 证伪 |
| 周期权重重排 | ③ | 单周期 IC 排序 ≠ 组合收益排序，证伪 |
| 置信度语义 | ④ | 高置信 ≠ edge 信号（IC ≈ 0 ~ −0.09） |
| 基准相对净超额 | ⑤ | 信号无净超额（随机子集可达） |
| 特征集×模型族消融 | ⑥ | 无一子集净超额推正 |
| 状态分层净超额 | ⑦/⑧ | 各状态均为负，`conditional_edge_hint=False` |
| 风险预测力 | ⑧ | 朴素基线 IC 0.69~0.75，模型增量仅 0.03~0.09，`no_risk_increment` |
| 全池基线冻结 | ⑨ | 26 标的净超额 −0.040%（t −0.41），`no_edge` |

**正式约定**：
1. TrendCast 定位为 **只读观测 / 风险预警**，不按「高置信」放大仓位；
2. 周期权重重排与采纳口径调整均 **不做**；
3. 方向预测这条线 **到顶**，不再加特征/换模型；
4. 风险预警用 **朴素 trailing-vol 基线** 即可，不必为模型输出立项；
5. 上述约定写入下游 `decision-feed/1` 契约的 `position_role = observer` 纪律（已就位）。

**不关闭的部分**：信号区分度弱（AUC ≈ 0.50~0.54）作为**已知限制**保留，
是否在 K 轮继续排查属人工决策（见 ROADMAP K 轮方向）。
## 八、下游 `_aggregate` 对齐指南（2026-10-08）

**问题**：下游 TradingView 仓自建 `trendcast_signal_source._aggregate` 使用
`short 0.2 / mid 0.5 / long 0.3` 权重 + `up_prob > 0.6 → BUY` / `< 0.4 → SELL` 硬编码阈值，
与 16_ 侧契约出口的 `0.30 / 0.35 / 0.35` 权重 + `advisory.recommended_threshold = 0.20` 不一致。
同一份预测在不同链路上被翻译成不同口径。

**替换映射**：

| 下游自建（旧） | 契约字段（新） | 说明 |
|---|---|---|
| `_aggregate(probs, weights={0.2,0.5,0.3})` | `feed.aggregate.composite_score` | 净看涨概率 ∈ [0,1]，权重已由生产端归一 |
| 自行 `score * 2 - 1` | `feed.aggregate.composite_signed` | ∈ [-1,1]，正=看多 |
| `up_prob > 0.6 → BUY` | `feed.aggregate.composite_score >= feed.advisory_config.recommended_threshold` | 门槛由生产端出口，非硬编码 |
| `< 0.4 → SELL` | `feed.aggregate.composite_score < (1 - feed.advisory_config.recommended_threshold)` | 对称下限 |
| `direction == "看涨" ? p : 1-p` | `feed.horizons.<h>.net_up_probability` | 逐周期净看涨概率，无需判断方向字符串 |
| `abs(up_prob - 0.5) * 2` | `feed.horizons.<h>.uncertainty` 或 `1 - abs(composite_signed)` | 命名的置信度/不确定度字段 |
| 自行权重硬编码 | `feed.aggregate.weights_used` / `feed.aggregate.weights_config` | 权重显式随附，可审计 |

**迁移步骤**：

1. **消费契约**：用 `get_decision_feed()`（已交付，2026-10-03）获取 `decision-feed/1` 契约；
2. **替换聚合**：删除 `_aggregate` 函数，直接读 `feed.aggregate.composite_score` / `composite_signed`；
3. **替换阈值**：删除 `0.6/0.4` 硬编码，改读 `feed.advisory_config.recommended_threshold`；
4. **替换方向判断**：删除 `direction == "看涨"` 逻辑，直接用 `net_up_probability`；
5. **验证**：对比迁移前后信号输出，确认 `composite_score` 与原 `_aggregate` 输出的差异仅来自权重口径（0.30/0.35/0.35 vs 0.2/0.5/0.3），非 bug。

**注意事项**：
- `position_role = observer` / `affects_gate = false` — 信号仅作只读观测，不作为仓位依据（§七约定）；
- `advisory_consumable` 0/38 — 当前全部信号未达采纳门槛，这是**如实挡住**而非缺陷；
- `coverage < 1.0` 时表示有周期缺失，`missing_horizons` 列出缺失项 — 不补 0.5；
- 权重口径变更（0.2/0.5/0.3 → 0.30/0.35/0.35）会改变信号输出 — 这是**对齐**而非 bug，
  原权重是下游自定、无依据；新权重由生产端出口、可审计。
