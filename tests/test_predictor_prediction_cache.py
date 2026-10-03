"""PredictionEngine 预测缓存守卫（src/inference/predictor.PredictionEngine.predict）。

背景（2026-10-03，下游决策源对接实测）：数据管线加重后单周期推理 ~5.4s、
26 标的全池 feed ≈ 7 分钟，消费方 10s 客户端超时打不通正路径；同进程重复
请求（重试 / 多端点复用）此前完全无缓存（同调用两次 16.7s）。

缓存契约（本组测试钉死）：
  - 键 = (symbol, horizon, 数据最后日期, 数据行数) —— 数据指纹入键，
    新数据入库后键必然变化，**结构上不可能返回过期预测**；
  - 只缓存成功读数；error 结果不缓存（数据/模型修复后下次调用自动重算）；
  - 键构造失败（date 列缺失等）⇒ 不缓存不报错，正确性不受影响；
  - 容量护栏：超过 PRED_CACHE_MAX 全清，行为退化为「不缓存」而非出错。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.inference.predictor as predictor_mod  # noqa: E402
from src.inference.predictor import PredictionEngine  # noqa: E402


def _make_df(dates: list[str], rows_extra: int = 0) -> pd.DataFrame:
    dates = list(dates)
    for i in range(rows_extra):
        y, m, d = map(int, dates[-1].split("-"))
        dates.append(f"{y:04d}-{m:02d}-{d + 1 + i:02d}")
    n = len(dates)
    return pd.DataFrame(
        {"date": dates, "close": [10.0 + i * 0.1 for i in range(n)]}
    )


class _StubCollector:
    def __init__(self, df: pd.DataFrame):
        self._df = df
        self.load_calls = 0

    def load_cached(self, symbol: str):
        self.load_calls += 1
        return self._df


@pytest.fixture()
def engine(monkeypatch):
    """不触网/不加载真实模型的引擎：models/feature_engineer/取数全部打桩。"""
    eng = PredictionEngine.__new__(PredictionEngine)
    eng.horizons = {"short_term": 5}
    eng.models = {"short_term_5d": object()}
    eng._pred_cache = {}
    eng._refresh_attempted = set()
    eng.config = {}
    df = _make_df(["2026-09-28", "2026-09-29", "2026-09-30"])
    collector = _StubCollector(df)
    monkeypatch.setattr(predictor_mod, "DataCollector", lambda cfg: collector)
    monkeypatch.setattr(
        eng, "_ensure_fresh", lambda symbol, df_, col: df_, raising=False
    )
    calls = {"n": 0}

    def _stub_predict_with_df(symbol, horizon_name, horizon_days, model_key, df_):
        calls["n"] += 1
        return {
            "symbol": symbol,
            "horizon": horizon_name,
            "probability": 0.6,
            "compute_seq": calls["n"],
        }

    monkeypatch.setattr(eng, "_predict_with_df", _stub_predict_with_df)
    eng._stub_calls = calls
    return eng


def test_same_data_second_call_hits_cache(engine):
    r1 = engine.predict("510300.SH", "short_term")
    r2 = engine.predict("510300.SH", "short_term")
    assert engine._stub_calls["n"] == 1          # 计算体只跑一次
    assert r1 == r2                              # 读数逐字段一致
    assert r2["compute_seq"] == 1


def test_new_row_appended_cache_invalidates(engine, monkeypatch):
    engine.predict("510300.SH", "short_term")
    # 追加一行新数据（行数变化 ⇒ 键变化）
    df2 = _make_df(["2026-09-28", "2026-09-29", "2026-09-30"], rows_extra=1)
    monkeypatch.setattr(
        predictor_mod, "DataCollector", lambda cfg: _StubCollector(df2)
    )
    r2 = engine.predict("510300.SH", "short_term")
    assert engine._stub_calls["n"] == 2          # 新数据必须重算
    assert r2["compute_seq"] == 2


def test_same_rows_new_date_cache_invalidates(engine, monkeypatch):
    engine.predict("510300.SH", "short_term")
    # 行数相同但最后日期不同（错位/修订场景）⇒ 键必须不同
    df2 = _make_df(["2026-09-29", "2026-09-30", "2026-10-09"])
    monkeypatch.setattr(
        predictor_mod, "DataCollector", lambda cfg: _StubCollector(df2)
    )
    engine.predict("510300.SH", "short_term")
    assert engine._stub_calls["n"] == 2


def test_error_result_is_not_cached(engine, monkeypatch):
    calls = engine._stub_calls

    def _fail_once(symbol, horizon_name, horizon_days, model_key, df_):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"error": "临时失败"}
        return {"symbol": symbol, "probability": 0.6}

    monkeypatch.setattr(engine, "_predict_with_df", _fail_once)
    r1 = engine.predict("510300.SH", "short_term")
    assert "error" in r1
    r2 = engine.predict("510300.SH", "short_term")   # error 不缓存 ⇒ 第二次重算
    assert "error" not in r2
    assert calls["n"] == 2


def test_missing_date_column_skips_cache_without_error(engine, monkeypatch):
    df_nodate = pd.DataFrame({"close": [10.0, 10.5, 11.0]})
    monkeypatch.setattr(
        predictor_mod, "DataCollector", lambda cfg: _StubCollector(df_nodate)
    )
    r1 = engine.predict("510300.SH", "short_term")
    r2 = engine.predict("510300.SH", "short_term")
    assert engine._stub_calls["n"] == 2          # 无指纹 ⇒ 不缓存，但照常出读数
    assert "error" not in r1 and "error" not in r2


def test_capacity_guard_clears_instead_of_growing_unbounded(engine):
    engine.PRED_CACHE_MAX = PredictionEngine.PRED_CACHE_MAX
    # 直接塞满缓存，验证超过护栏后可继续工作（全清退化，不抛错）
    for i in range(PredictionEngine.PRED_CACHE_MAX + 1):
        engine._pred_cache[("S", "short_term", f"2026-01-{(i % 28) + 1:02d}", i)] = {
            "probability": 0.5
        }
    engine.predict("510300.SH", "short_term")
    assert len(engine._pred_cache) <= PredictionEngine.PRED_CACHE_MAX
    assert engine._stub_calls["n"] == 1


def test_model_missing_short_circuits_before_cache(engine):
    r = engine.predict("510300.SH", "mid_term")   # mid_term_10d 不在 models
    assert "error" in r
    assert engine._stub_calls["n"] == 0
