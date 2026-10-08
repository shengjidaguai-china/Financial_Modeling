"""池级组合回测器全池口径（S28 / K2）：38 标的 × 三臂全池组合回测。

本模块做什么：
  - 对全池 38 标的构建三臂（等权/ATR 倒数/置信度）权重计划；
  - 复用 S21 引擎逐臂 × 逐成本档跑组合回测；
  - 输出跨标的聚合读数 + 三臂对照统计；
  - 结论不管好坏都如实入库。

本模块不做什么：
  - 不改门禁：affects_gate 恒为 False；
  - 不改引擎：复用 run_portfolio_backtest，零改动；
  - 不猜：样本不足时如实标 unavailable。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.eval.portfolio_backtest import (
    build_atr_inverse_plan,
    build_confidence_plan,
    build_equal_weight_plan,
    run_portfolio_backtest,
    WEIGHT_ARMS,
)

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def _build_full_pool_signal_frames(
    price_frames: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """全池标的每日激活信号（BUY = 全程持有，等权 buy-and-hold 基线）。"""
    signal_frames: Dict[str, pd.DataFrame] = {}
    for sym, df in price_frames.items():
        if "date" not in df.columns or "close" not in df.columns:
            continue
        n = len(df)
        signal_frames[sym] = pd.DataFrame({
            "date": df["date"].to_numpy(),
            "action": ["BUY"] * n,
            "confidence": [1.0] * n,
        })
    return signal_frames


def _build_all_arms(
    price_frames: Dict[str, pd.DataFrame],
    signal_frames: Dict[str, pd.DataFrame],
    atr_window: int = 14,
) -> Dict[str, tuple]:
    """构建三臂（权重计划, normalize 模式）。"""
    equal_plans = build_equal_weight_plan(signal_frames)
    arms: Dict[str, tuple] = {
        "equal": (equal_plans, "equal_active"),
        "atr_inverse": (
            build_atr_inverse_plan(price_frames, equal_plans, atr_window=atr_window),
            None,
        ),
        "confidence": (
            build_confidence_plan(signal_frames, equal_plans),
            None,
        ),
    }
    return arms


def _aggregate_arm_stats(
    arm_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """跨成本档聚合单臂统计。"""
    tiers = ["conservative", "base", "aggressive"]
    out: Dict[str, Any] = {}
    for tier in tiers:
        r = arm_results.get(tier, {})
        if not r.get("available"):
            out[tier] = {"available": False, "reason": r.get("reason", "unknown")}
            continue
        m = r.get("metrics", {})
        out[tier] = {
            "available": True,
            "total_return": m.get("total_return"),
            "annualized_return": m.get("annualized_return"),
            "sharpe": m.get("sharpe"),
            "max_drawdown": m.get("max_drawdown"),
            "total_turnover": m.get("total_turnover"),
            "cost_drag": m.get("cost_drag"),
            "n_days": m.get("n_days"),
        }
    return out


def _cross_arm_comparison(
    arms_stats: Dict[str, Dict[str, Any]],
    base_tier: str = "base",
) -> Dict[str, Any]:
    """三臂对照：以等权为基准，计算各臂超额。"""
    base_arm = arms_stats.get("equal", {}).get(base_tier, {})
    if not base_arm.get("available"):
        return {"available": False, "reason": "等权基准不可用"}

    base_return = base_arm.get("annualized_return", 0.0) or 0.0
    base_sharpe = base_arm.get("sharpe", 0.0) or 0.0

    comparison: Dict[str, Any] = {"available": True, "base_tier": base_tier}
    for arm in WEIGHT_ARMS:
        stats = arms_stats.get(arm, {}).get(base_tier, {})
        if not stats.get("available"):
            comparison[arm] = {"available": False}
            continue
        arm_return = stats.get("annualized_return", 0.0) or 0.0
        arm_sharpe = stats.get("sharpe", 0.0) or 0.0
        comparison[arm] = {
            "available": True,
            "annualized_return": arm_return,
            "sharpe": arm_sharpe,
            "return_excess_vs_equal": arm_return - base_return,
            "sharpe_excess_vs_equal": arm_sharpe - base_sharpe,
        }
    return comparison


def build_report(
    price_frames: Dict[str, pd.DataFrame],
    cost_tiers: Optional[Sequence[Dict[str, Any]]] = None,
    atr_window: int = 14,
) -> Dict[str, Any]:
    """全池组合回测基线报告。

    Parameters
    ----------
    price_frames : {symbol: DataFrame} 全池行情（date/open/high/low/close/volume）
    cost_tiers : 成本档列表（缺省用 T11.2 三档）
    atr_window : ATR 窗口（缺省 14）
    """
    from src.eval.factor_metrics import T112_COST_TIERS

    if cost_tiers is None:
        cost_tiers = T112_COST_TIERS

    report: Dict[str, Any] = {
        "command": "pool-backtest-baseline",
        "kind": "pool_backtest_baseline",
        "affects_gate": False,
        "n_symbols_input": len(price_frames),
        "symbols": list(price_frames.keys()),
        "arms": list(WEIGHT_ARMS),
        "cost_tiers": [t["name"] for t in cost_tiers],
    }

    if len(price_frames) < 2:
        report["available"] = False
        report["reason"] = f"标的数不足（{len(price_frames)} < 2）"
        return report

    signal_frames = _build_full_pool_signal_frames(price_frames)
    if not signal_frames:
        report["available"] = False
        report["reason"] = "无法构建信号 frame"
        return report

    arms = {}
    try:
        arms = _build_all_arms(price_frames, signal_frames, atr_window=atr_window)
    except ValueError as e:
        logger.warning(f"[pool-backtest-baseline] 三臂构建失败（{e}），退化为等权")
        equal_plans = build_equal_weight_plan(signal_frames)
        arms = {"equal": (equal_plans, "equal_active")}

    arms_results: Dict[str, Dict[str, Any]] = {}
    for arm_name, (plans, normalize) in arms.items():
        if not plans:
            arms_results[arm_name] = {t["name"]: {"available": False, "reason": "空权重计划"} for t in cost_tiers}
            continue
        tier_results: Dict[str, Any] = {}
        for tier in cost_tiers:
            result = run_portfolio_backtest(
                price_frames,
                plans,
                cost_one_side_value=float(tier["one_side"]),
                normalize=normalize,
                trading_days=TRADING_DAYS,
            )
            tier_results[tier["name"]] = result
        arms_results[arm_name] = tier_results

    arms_stats = {arm: _aggregate_arm_stats(results) for arm, results in arms_results.items()}
    report["arms_stats"] = arms_stats
    report["cross_arm_comparison"] = _cross_arm_comparison(arms_stats)

    base_result = arms_results.get("equal", {}).get("base", {})
    report["available"] = base_result.get("available", False)
    report["n_symbols_used"] = len(signal_frames)
    if base_result.get("available"):
        report["date_range"] = base_result.get("date_range")

    report["per_arm_per_tier"] = {
        arm: {tier: _aggregate_arm_stats({tier: r})[tier] for tier, r in results.items()}
        for arm, results in arms_results.items()
    }

    return report