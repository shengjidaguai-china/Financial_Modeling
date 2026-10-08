"""S29 / K3 — 波动/回撤预警项目守卫。

验证 risk_alert 模块的核心契约：
  - 逐标的 trailing-vol + 回撤读数可用
  - 预警等级分类正确（normal/elevated/high/extreme）
  - 全池聚合：等级分布 + top_alerts
  - affects_gate 恒为 False
  - 无前视：只用截至 t 的数据
  - 样本不足时如实标 unavailable
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_price_frame(n: int = 300, seed: int = 42, vol: float = 0.01) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, vol, n)))
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


def _make_crash_frame(n: int = 300, crash_start: int = 250) -> pd.DataFrame:
    """前半段平稳，后半段暴跌（触发 high/extreme 回撤预警）。"""
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = np.full(n, 100.0)
    for i in range(1, n):
        if i < crash_start:
            close[i] = close[i - 1] * (1.0 + np.random.RandomState(i).normal(0, 0.005))
        else:
            close[i] = close[i - 1] * 0.98
    return pd.DataFrame({
        "date": dates,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 1e6,
    })


class TestComputeTrailingVol:
    def test_basic(self):
        from src.eval.risk_alert import compute_trailing_vol

        close = pd.Series(np.linspace(100, 110, 100))
        vol = compute_trailing_vol(close, window=20)
        assert len(vol) == 100
        assert vol.iloc[:19].isna().all()
        assert vol.iloc[20:].notna().all()

    def test_no_front_look(self):
        from src.eval.risk_alert import compute_trailing_vol

        close_a = pd.Series(np.linspace(100, 110, 50).tolist() + [200] * 50)
        close_b = pd.Series(np.linspace(100, 110, 50).tolist() + [100] * 50)
        vol_a = compute_trailing_vol(close_a, window=20)
        vol_b = compute_trailing_vol(close_b, window=20)
        assert vol_a.iloc[49] == vol_b.iloc[49]


class TestComputeDrawdown:
    def test_basic(self):
        from src.eval.risk_alert import compute_drawdown

        close = pd.Series([100, 105, 110, 90, 85, 95])
        dd = compute_drawdown(close, window=6)
        assert dd.iloc[0] == pytest.approx(0.0)
        assert dd.iloc[2] == pytest.approx(0.0)
        assert dd.iloc[4] < 0
        assert dd.iloc[4] == pytest.approx(85.0 / 110.0 - 1.0, rel=1e-6)

    def test_no_front_look(self):
        from src.eval.risk_alert import compute_drawdown

        close_a = pd.Series(list(range(100, 150)) + [200] * 50)
        close_b = pd.Series(list(range(100, 150)) + [100] * 50)
        dd_a = compute_drawdown(close_a, window=20)
        dd_b = compute_drawdown(close_b, window=20)
        assert dd_a.iloc[49] == dd_b.iloc[49]


class TestBuildSymbolAlert:
    def test_available_with_sufficient_data(self):
        from src.eval.risk_alert import build_symbol_alert

        close = _make_price_frame(300)["close"]
        result = build_symbol_alert(close)
        assert result["available"] is True
        assert "trailing_vol_annualized" in result
        assert "current_drawdown" in result
        assert "vol_alert_level" in result
        assert "drawdown_alert_level" in result
        assert "overall_alert_level" in result

    def test_unavailable_with_insufficient_data(self):
        from src.eval.risk_alert import build_symbol_alert

        close = pd.Series(np.linspace(100, 110, 20))
        result = build_symbol_alert(close)
        assert result["available"] is False
        assert "reason" in result

    def test_alert_levels_in_valid_set(self):
        from src.eval.risk_alert import ALERT_LEVELS, build_symbol_alert

        close = _make_price_frame(300)["close"]
        result = build_symbol_alert(close)
        assert result["vol_alert_level"] in ALERT_LEVELS
        assert result["drawdown_alert_level"] in ALERT_LEVELS
        assert result["overall_alert_level"] in ALERT_LEVELS

    def test_overall_is_max_of_vol_and_dd(self):
        from src.eval.risk_alert import ALERT_LEVELS, build_symbol_alert

        close = _make_price_frame(300)["close"]
        result = build_symbol_alert(close)
        vol_rank = ALERT_LEVELS.index(result["vol_alert_level"])
        dd_rank = ALERT_LEVELS.index(result["drawdown_alert_level"])
        overall_rank = ALERT_LEVELS.index(result["overall_alert_level"])
        assert overall_rank == max(vol_rank, dd_rank)

    def test_crash_triggers_high_alert(self):
        from src.eval.risk_alert import ALERT_LEVELS, build_symbol_alert

        close = _make_crash_frame(300, crash_start=250)["close"]
        result = build_symbol_alert(close, drawdown_window=60)
        assert result["available"] is True
        dd_rank = ALERT_LEVELS.index(result["drawdown_alert_level"])
        assert dd_rank >= 2  # high or extreme


class TestBuildReport:
    def test_report_structure(self):
        from src.eval.risk_alert import build_report

        pool = _make_pool(5, 300)
        report = build_report(pool)

        assert report["kind"] == "risk_alert"
        assert report["command"] == "risk-alert"
        assert report["affects_gate"] is False
        assert report["readonly"] is True
        assert "alert_level_distribution" in report
        assert "top_alerts" in report
        assert "per_symbol" in report

    def test_all_symbols_covered(self):
        from src.eval.risk_alert import build_report

        pool = _make_pool(8, 300)
        report = build_report(pool)
        assert report["n_symbols_input"] == 8
        assert len(report["per_symbol"]) == 8

    def test_level_distribution_sums_to_available(self):
        from src.eval.risk_alert import build_report

        pool = _make_pool(5, 300)
        report = build_report(pool)
        dist = report["alert_level_distribution"]
        total = sum(dist.values())
        assert total == report["n_symbols_available"]

    def test_top_alerts_sorted_by_severity(self):
        from src.eval.risk_alert import ALERT_LEVELS, build_report

        pool = _make_pool(10, 300)
        report = build_report(pool)
        alerts = report["top_alerts"]
        for i in range(len(alerts) - 1):
            r0 = ALERT_LEVELS.index(alerts[i]["overall_alert_level"])
            r1 = ALERT_LEVELS.index(alerts[i + 1]["overall_alert_level"])
            assert r0 >= r1

    def test_top_alerts_limit(self):
        from src.eval.risk_alert import build_report

        pool = _make_pool(20, 300)
        report = build_report(pool)
        assert len(report["top_alerts"]) <= 10

    def test_empty_pool(self):
        from src.eval.risk_alert import build_report

        report = build_report({})
        assert report["available"] is False
        assert report["n_symbols_input"] == 0

    def test_missing_close_column(self):
        from src.eval.risk_alert import build_report

        pool = {"BAD": pd.DataFrame({"date": [1, 2, 3], "open": [1, 2, 3]})}
        report = build_report(pool)
        assert report["per_symbol"]["BAD"]["available"] is False

    def test_summary_present(self):
        from src.eval.risk_alert import build_report

        pool = _make_pool(5, 300)
        report = build_report(pool)
        assert "summary" in report
        assert isinstance(report["summary"], str)
        assert len(report["summary"]) > 0

    def test_boundary_declares_readonly(self):
        from src.eval.risk_alert import build_report

        pool = _make_pool(3, 300)
        report = build_report(pool)
        assert "只读" in report["boundary"]
        assert "affects_gate" not in report["boundary"].lower() or "不改" in report["boundary"]