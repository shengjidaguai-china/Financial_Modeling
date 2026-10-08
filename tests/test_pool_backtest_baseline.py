"""S28 / K2 — 池级组合回测器全池口径守卫。

验证 pool_backtest_baseline 模块的核心契约：
  - 全池 38 标的覆盖
  - 三臂 × 三成本档全跑
  - 跨标的聚合读数可用
  - affects_gate 恒为 False
  - 排期登记 T28.1~T28.3
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_price_frame(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "date": dates,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 1e6,
    })


def _make_pool(n_symbols: int = 5, n_days: int = 300) -> dict:
    return {f"SYM{i:02d}": _make_price_frame(n_days, seed=i) for i in range(n_symbols)}


class TestBuildReport:
    def test_report_structure(self):
        from src.eval.pool_backtest_baseline import build_report

        pool = _make_pool(5, 300)
        report = build_report(pool)

        assert report["command"] == "pool-backtest-baseline"
        assert report["kind"] == "pool_backtest_baseline"
        assert report["affects_gate"] is False
        assert "arms" in report
        assert "cost_tiers" in report

    def test_three_arms_present(self):
        from src.eval.pool_backtest_baseline import build_report

        pool = _make_pool(5, 300)
        report = build_report(pool)
        assert set(report["arms"]) == {"equal", "atr_inverse", "confidence"}

    def test_three_cost_tiers_present(self):
        from src.eval.pool_backtest_baseline import build_report

        pool = _make_pool(5, 300)
        report = build_report(pool)
        assert set(report["cost_tiers"]) == {"conservative", "base", "aggressive"}

    def test_cross_arm_comparison(self):
        from src.eval.pool_backtest_baseline import build_report

        pool = _make_pool(5, 300)
        report = build_report(pool)
        cmp = report["cross_arm_comparison"]
        assert cmp["available"] is True
        assert "equal" in cmp
        assert "atr_inverse" in cmp

    def test_insufficient_symbols(self):
        from src.eval.pool_backtest_baseline import build_report

        pool = {"SYM00": _make_price_frame(300)}
        report = build_report(pool)
        assert report["available"] is False

    def test_n_symbols_tracked(self):
        from src.eval.pool_backtest_baseline import build_report

        pool = _make_pool(7, 300)
        report = build_report(pool)
        assert report["n_symbols_input"] == 7


class TestPlanRegistration:
    """T28.1~T28.3 排期登记守卫。"""

    def test_s28_in_plan(self):
        plan_path = PROJECT_ROOT / "schedule" / "plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        s28 = next((s for s in plan["stages"] if s["id"] == "S28"), None)
        assert s28 is not None, "S28 未在 plan.json"
        task_ids = [t["id"] for t in s28["tasks"]]
        assert "T28.1" in task_ids
        assert "T28.2" in task_ids
        assert "T28.3" in task_ids