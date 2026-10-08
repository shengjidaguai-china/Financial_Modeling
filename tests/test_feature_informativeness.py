"""S27 / K1 — 特征信息量审计守卫。

验证 feature_informativeness 模块的核心契约：
  - MI/IC 计算正确（已知特征的排序合理）
  - 零信息特征被识别
  - 冗余对被识别
  - affects_gate 恒为 False
  - 排期登记 T27.1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_synthetic_frame(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """构造含信息特征 + 零信息特征 + 冗余特征的合成面板。"""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    df = pd.DataFrame({
        "date": dates,
        "open": close * (1 + rng.normal(0, 0.005, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.01, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.01, n))),
        "close": close,
        "volume": rng.uniform(1e6, 1e7, n),
    })
    fwd_ret_5 = df["close"].shift(-5) / df["close"] - 1
    df["target_5d"] = (fwd_ret_5 > 0).astype(float)
    df["target_5d"][fwd_ret_5.isna()] = np.nan
    df["f_informative"] = fwd_ret_5.shift(1) + rng.normal(0, 0.01, n)
    df["f_noise"] = rng.normal(0, 1, n)
    df["f_redundant"] = df["f_informative"] * 1.0 + rng.normal(0, 1e-6, n)
    return df


class TestEvaluateSymbol:
    def test_returns_available_for_valid_data(self):
        from src.eval.feature_informativeness import evaluate_symbol

        df = _make_synthetic_frame()
        result = evaluate_symbol(df, ["f_informative", "f_noise", "f_redundant"], 5)
        assert result["available"] is True
        assert result["symbol_samples"] > 0

    def test_mi_dict_has_all_features(self):
        from src.eval.feature_informativeness import evaluate_symbol

        df = _make_synthetic_frame()
        cols = ["f_informative", "f_noise", "f_redundant"]
        result = evaluate_symbol(df, cols, 5)
        assert set(result["mi"].keys()) == set(cols)
        assert set(result["ic"].keys()) == set(cols)

    def test_insufficient_samples_marked_unavailable(self):
        from src.eval.feature_informativeness import evaluate_symbol

        df = _make_synthetic_frame(n=50)
        result = evaluate_symbol(df, ["f_informative"], 5)
        assert result["available"] is False
        assert "不足" in result["reason"]

    def test_missing_target_column(self):
        from src.eval.feature_informativeness import evaluate_symbol

        df = _make_synthetic_frame()
        result = evaluate_symbol(df, ["f_informative"], 10)
        assert result["available"] is False
        assert "target_10d" in result["reason"]


class TestComputeRedundancy:
    def test_finds_redundant_pair(self):
        from src.eval.feature_informativeness import compute_redundancy

        df = _make_synthetic_frame()
        pairs = compute_redundancy(df, ["f_informative", "f_redundant", "f_noise"])
        assert len(pairs) >= 1
        pair = pairs[0]
        assert {pair["a"], pair["b"]} == {"f_informative", "f_redundant"}
        assert pair["corr"] > 0.9

    def test_no_redundant_pair_for_independent_features(self):
        from src.eval.feature_informativeness import compute_redundancy

        df = _make_synthetic_frame()
        pairs = compute_redundancy(df, ["f_informative", "f_noise"])
        assert len(pairs) == 0


class TestBuildReport:
    def test_report_structure(self):
        from src.eval.feature_informativeness import build_report

        df = _make_synthetic_frame()
        cols = ["f_informative", "f_noise", "f_redundant"]
        report = build_report({"SYM1": df}, cols, horizons=[5])

        assert report["command"] == "feature-informativeness"
        assert report["kind"] == "feature_informativeness_audit"
        assert report["affects_gate"] is False
        assert report["n_features"] == 3
        assert "5d" in report["per_horizon"]

    def test_zero_info_features_identified(self):
        from src.eval.feature_informativeness import build_report

        df = _make_synthetic_frame()
        cols = ["f_informative", "f_noise", "f_redundant"]
        report = build_report({"SYM1": df}, cols, horizons=[5])
        h5 = report["per_horizon"]["5d"]
        assert "f_noise" in h5["zero_info_features"] or h5["mi_mean"]["f_noise"] < h5["mi_mean"]["f_informative"]

    def test_redundant_pairs_in_report(self):
        from src.eval.feature_informativeness import build_report

        df = _make_synthetic_frame()
        cols = ["f_informative", "f_redundant", "f_noise"]
        report = build_report({"SYM1": df}, cols, horizons=[5])
        assert len(report["redundant_pairs"]) >= 1


class TestFeatureExclusion:
    """T27.4 特征排除机制守卫。"""

    def test_config_has_feature_exclusions(self):
        import yaml
        config_path = PROJECT_ROOT / "configs" / "config_pro.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        exclusions = config.get("features", {}).get("feature_exclusions", [])
        assert isinstance(exclusions, list) and len(exclusions) > 0, (
            "feature_exclusions 未配置或为空"
        )

    def test_excluded_features_not_in_feature_columns(self):
        import yaml
        from src.data.preprocessor import FeatureEngineer

        config = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "config_pro.yaml").read_text(encoding="utf-8")
        )
        fe = FeatureEngineer(config)
        exclusions = set(config["features"]["feature_exclusions"])
        df = pd.DataFrame({
            "date": [1], "close": [100], "open": [100], "high": [100],
            "low": [100], "volume": [1000],
            "rsi": [50], "atr": [5], "vwap": [100], "target_5d": [1],
        })
        cols = fe.get_feature_columns(df, 5)
        for ex in ["rsi", "atr"]:
            assert ex not in cols, f"{ex} 应被排除"

    def test_exclusion_count_matches_audit(self):
        import yaml
        config_path = PROJECT_ROOT / "configs" / "config_pro.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        exclusions = config["features"]["feature_exclusions"]
        assert len(exclusions) == 35, f"排除列表应为 35 个，实际 {len(exclusions)}"


class TestPlanRegistration:
    """T27.1 排期登记守卫。"""

    def test_t271_in_plan(self):
        plan_path = PROJECT_ROOT / "schedule" / "plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        s27 = next((s for s in plan["stages"] if s["id"] == "S27"), None)
        assert s27 is not None, "S27 未在 plan.json"
        t271 = next((t for t in s27["tasks"] if t["id"] == "T27.1"), None)
        assert t271 is not None, "T27.1 未在 S27 tasks"
        assert "feature_informativeness" in t271["name"].lower() or "信息量" in t271["name"]