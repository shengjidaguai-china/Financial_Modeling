"""波动/回撤预警项目（S29 / K3）：朴素 trailing-vol 基线产品化。

## 背景

Issue #55 九轮排查一致结论：方向预测到顶，TrendCast 定位为**只读观测 / 风险预警**。
风险预测力检验（`risk_signal.py`）已验证朴素 trailing-vol 基线对未来波动 IC 0.69~0.75，
**模型输出不提供超出它的增量** ⇒ 用朴素基线即可，不必为模型输出立项。

本模块把那条已验证的朴素基线**产品化**为可读的预警系统：

  - 逐标的 trailing 波动率 + 历史分位数 → 波动预警等级
  - 逐标的当前回撤（相对 trailing high）→ 回撤预警等级
  - 全池聚合：预警等级分布 + 高预警标的清单

## 纪律

  - **纯价格驱动**：只用 close 列，不碰特征 / 模型输出 / 门禁。
  - **无前视**：所有读数只用截至 t 的数据，trailing window 不含未来。
  - **affects_gate 恒 False**：只读预警，不改门禁 / 权重 / 池 / 配置。
  - **零新依赖**：只用 numpy / pandas。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252

DEFAULT_VOL_WINDOW = 20
DEFAULT_DRAWDOWN_WINDOW = 60
DEFAULT_VOL_PERCENTILES = (50, 75, 90)
DEFAULT_DRAWDOWN_THRESHOLDS = (-0.05, -0.10, -0.20)

ALERT_LEVELS = ("normal", "elevated", "high", "extreme")


def _finite(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _round(v: Any, nd: int = 6) -> Optional[float]:
    f = _finite(v)
    return None if f is None else round(f, nd)


def _classify_by_percentile(value: float, thresholds: Tuple[float, ...]) -> str:
    """按分位数阈值分类：低于 thresholds[0] → normal，…，高于最后一个 → extreme。"""
    for i, t in enumerate(thresholds):
        if value < t:
            return ALERT_LEVELS[i]
    return ALERT_LEVELS[-1]


def _classify_drawdown(dd: float, thresholds: Tuple[float, ...]) -> str:
    """回撤分类：dd 是负值，越深越负。thresholds 从浅到深（如 -0.05, -0.10, -0.20）。"""
    for i, t in enumerate(thresholds):
        if dd > t:
            return ALERT_LEVELS[i]
    return ALERT_LEVELS[-1]


def _alert_rank(level: str) -> int:
    """预警等级 → 整数序（用于排序 / 聚合）。"""
    try:
        return ALERT_LEVELS.index(level)
    except ValueError:
        return 0


def compute_trailing_vol(close: pd.Series, window: int = DEFAULT_VOL_WINDOW) -> pd.Series:
    """trailing 波动率（年化）：截至 t 的 window 日日收益标准差 × sqrt(252)。

    无前视：只用 [t-window, t] 的收益。
    """
    c = pd.to_numeric(close, errors="coerce").astype(float)
    ret = c.pct_change()
    return ret.rolling(int(window)).std() * np.sqrt(TRADING_DAYS)


def compute_drawdown(close: pd.Series, window: int = DEFAULT_DRAWDOWN_WINDOW) -> pd.Series:
    """当前回撤：相对 trailing window 内最高价的回撤（负值）。

    dd_t = close_t / max(close[t-window..t]) - 1 ≤ 0。
    无前视：只用 [t-window, t] 的价格。
    """
    c = pd.to_numeric(close, errors="coerce").astype(float)
    rolling_high = c.rolling(int(window), min_periods=1).max()
    return (c / rolling_high) - 1.0


def _percentile_thresholds(
    series: pd.Series, percentiles: Tuple[int, ...]
) -> Tuple[float, ...]:
    """从历史分布算分位数阈值（无前视：用全序列的 nanquantile）。"""
    arr = series.to_numpy(dtype=float)
    valid = arr[np.isfinite(arr)]
    if len(valid) < 20:
        return tuple()
    return tuple(float(np.nanquantile(valid, p / 100.0)) for p in percentiles)


def build_symbol_alert(
    close: pd.Series,
    vol_window: int = DEFAULT_VOL_WINDOW,
    drawdown_window: int = DEFAULT_DRAWDOWN_WINDOW,
    vol_percentiles: Tuple[int, ...] = DEFAULT_VOL_PERCENTILES,
    drawdown_thresholds: Tuple[float, ...] = DEFAULT_DRAWDOWN_THRESHOLDS,
) -> Dict[str, Any]:
    """逐标的预警读数：trailing vol + 回撤 + 预警等级。

    返回最新一期（最后一行）的读数 + 历史分位数阈值。
    """
    c = pd.to_numeric(close, errors="coerce").astype(float)
    n = int(c.notna().sum())
    if n < vol_window + 10:
        return {"available": False, "reason": f"样本不足（{n} < {vol_window + 10}）"}

    trail_vol = compute_trailing_vol(c, vol_window)
    drawdown = compute_drawdown(c, drawdown_window)

    vol_thresholds = _percentile_thresholds(trail_vol, vol_percentiles)
    if not vol_thresholds:
        return {"available": False, "reason": "波动率历史分布不足"}

    latest_idx = c.index[-1]
    latest_vol = _finite(trail_vol.iloc[-1])
    latest_dd = _finite(drawdown.iloc[-1])
    latest_close = _finite(c.iloc[-1])

    if latest_vol is None or latest_dd is None:
        return {"available": False, "reason": "最新一期波动率或回撤不可用"}

    vol_level = _classify_by_percentile(latest_vol, vol_thresholds)
    dd_level = _classify_drawdown(latest_dd, drawdown_thresholds)
    overall_rank = max(_alert_rank(vol_level), _alert_rank(dd_level))
    overall_level = ALERT_LEVELS[overall_rank]

    return {
        "available": True,
        "latest_close": _round(latest_close, 4),
        "trailing_vol_annualized": _round(latest_vol, 4),
        "vol_percentile_thresholds": {
            f"p{p}": _round(t, 4) for p, t in zip(vol_percentiles, vol_thresholds)
        },
        "vol_alert_level": vol_level,
        "current_drawdown": _round(latest_dd, 4),
        "drawdown_thresholds": {
            ALERT_LEVELS[i + 1]: _round(t, 4)
            for i, t in enumerate(drawdown_thresholds)
        },
        "drawdown_alert_level": dd_level,
        "overall_alert_level": overall_level,
        "n_samples": n,
    }


def build_report(
    data: Dict[str, pd.DataFrame],
    vol_window: int = DEFAULT_VOL_WINDOW,
    drawdown_window: int = DEFAULT_DRAWDOWN_WINDOW,
    vol_percentiles: Tuple[int, ...] = DEFAULT_VOL_PERCENTILES,
    drawdown_thresholds: Tuple[float, ...] = DEFAULT_DRAWDOWN_THRESHOLDS,
) -> Dict[str, Any]:
    """全池波动/回撤预警报告。

    输入：price_frames（symbol → DataFrame with date/close 列）。
    输出：逐标的预警读数 + 全池聚合。
    """
    per_symbol: Dict[str, Any] = {}
    available_symbols: List[str] = []

    for symbol, px in (data or {}).items():
        if "close" not in (px.columns if hasattr(px, "columns") else []):
            per_symbol[symbol] = {"available": False, "reason": "缺 close 列"}
            continue
        reading = build_symbol_alert(
            px["close"],
            vol_window=vol_window,
            drawdown_window=drawdown_window,
            vol_percentiles=vol_percentiles,
            drawdown_thresholds=drawdown_thresholds,
        )
        per_symbol[symbol] = reading
        if reading.get("available"):
            available_symbols.append(symbol)

    level_counts: Dict[str, int] = {lvl: 0 for lvl in ALERT_LEVELS}
    for sym in available_symbols:
        lvl = per_symbol[sym]["overall_alert_level"]
        level_counts[lvl] = level_counts.get(lvl, 0) + 1

    elevated_symbols = sorted(
        available_symbols,
        key=lambda s: _alert_rank(per_symbol[s]["overall_alert_level"]),
        reverse=True,
    )
    top_alerts = [
        {
            "symbol": s,
            "overall_alert_level": per_symbol[s]["overall_alert_level"],
            "vol_alert_level": per_symbol[s]["vol_alert_level"],
            "drawdown_alert_level": per_symbol[s]["drawdown_alert_level"],
            "trailing_vol_annualized": per_symbol[s]["trailing_vol_annualized"],
            "current_drawdown": per_symbol[s]["current_drawdown"],
        }
        for s in elevated_symbols[:10]
    ]

    n_elevated_or_higher = sum(
        level_counts.get(lvl, 0) for lvl in ("elevated", "high", "extreme")
    )

    report: Dict[str, Any] = {
        "kind": "risk_alert",
        "command": "risk-alert",
        "affects_gate": False,
        "readonly": True,
        "boundary": "只读预警：不改门禁 / 权重 / 池 / 配置。纯价格驱动，无模型输出。",
        "n_symbols_input": int(len(data or {})),
        "n_symbols_available": int(len(available_symbols)),
        "vol_window": int(vol_window),
        "drawdown_window": int(drawdown_window),
        "vol_percentiles": list(vol_percentiles),
        "drawdown_thresholds": {
            ALERT_LEVELS[i + 1]: _round(t, 4)
            for i, t in enumerate(drawdown_thresholds)
        },
        "alert_level_distribution": level_counts,
        "n_elevated_or_higher": int(n_elevated_or_higher),
        "top_alerts": top_alerts,
        "per_symbol": per_symbol,
    }

    if not available_symbols:
        report["available"] = False
        report["reason"] = "无可用标的（样本不足或缺 close 列）"
        report["summary"] = "预警不可用：全池无可用读数。"
        return report

    report["available"] = True
    if level_counts.get("extreme", 0) > 0:
        report["summary"] = (
            f"{level_counts['extreme']} 个标的处于 extreme 预警，"
            f"{n_elevated_or_higher} 个标的 elevated 或更高。"
        )
    elif level_counts.get("high", 0) > 0:
        report["summary"] = (
            f"{level_counts['high']} 个标的处于 high 预警，"
            f"{n_elevated_or_higher} 个标的 elevated 或更高。"
        )
    elif level_counts.get("elevated", 0) > 0:
        report["summary"] = (
            f"{level_counts['elevated']} 个标的处于 elevated 预警，"
            "无 high/extreme。"
        )
    else:
        report["summary"] = "全池正常，无标的处于 elevated 或更高预警。"
    return report