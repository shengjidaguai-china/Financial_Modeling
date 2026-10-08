"""特征信息量审计（S27 / K1）：52 指标 MI/IC 排序 + 冗余诊断。

本模块做什么：
  - 对全池标的计算每个特征列的互信息（MI）与信息系数（IC）；
  - 按 MI / |IC| 排序，识别零信息特征（MI ≈ 0）与冗余特征（高相关对）；
  - 结论不管好坏都如实入库。

本模块不做什么：
  - 不改门禁：affects_gate 恒为 False，只产出证据；
  - 不删特征：是否缩池/删特征由人工检查点（T27.4）决定；
  - 不猜：样本不足时如实标 unavailable。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_SAMPLES = 200
MI_ZERO_THRESHOLD = 0.001
REDUNDANCY_CORR_THRESHOLD = 0.90


def _safe_mi(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """计算每个特征对标签的互信息；失败返回全 0。"""
    try:
        from sklearn.feature_selection import mutual_info_classif
        return mutual_info_classif(X, y, random_state=42)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[feature-informativeness] MI 计算失败: {e}")
        return np.zeros(X.shape[1], dtype=float)


def _spearman_ic(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman 秩相关；样本不足或方差为 0 时返回 0.0。"""
    from src.inference.ic import spearman_ic
    return spearman_ic(a, b)


def evaluate_symbol(
    df_features: pd.DataFrame,
    feature_cols: List[str],
    horizon_days: int,
) -> Dict[str, Any]:
    """单标的特征信息量评估。

    Parameters
    ----------
    df_features : 已含特征列与 target_{h}d 列的 DataFrame
    feature_cols : 特征列名列表
    horizon_days : 预测周期（5/10/20）
    """
    target_col = f"target_{horizon_days}d"
    out: Dict[str, Any] = {"available": False, "symbol_samples": 0}

    if target_col not in df_features.columns:
        out["reason"] = f"缺 {target_col} 列"
        return out

    keep_cols = list(set(feature_cols + [target_col] + ["close"]))
    valid = df_features[keep_cols].dropna()
    n = len(valid)
    out["symbol_samples"] = n
    if n < MIN_SAMPLES:
        out["reason"] = f"样本不足（{n} < {MIN_SAMPLES}）"
        return out

    y = valid[target_col].to_numpy(dtype=float).astype(int)
    if len(np.unique(y)) < 2:
        out["reason"] = "标签单一类"
        return out

    X = valid[feature_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    mi_scores = _safe_mi(X, y)

    fwd_ret = valid["close"].shift(-horizon_days) / valid["close"] - 1
    fwd_ret = fwd_ret.reindex(valid.index)
    ic_scores: List[float] = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.std(col) < 1e-12:
            ic_scores.append(0.0)
            continue
        ic_scores.append(_spearman_ic(col.tolist(), fwd_ret.tolist()))

    out.update({
        "available": True,
        "mi": {feature_cols[j]: float(mi_scores[j]) for j in range(len(feature_cols))},
        "ic": {feature_cols[j]: float(ic_scores[j]) for j in range(len(feature_cols))},
    })
    return out


def compute_redundancy(
    df_features: pd.DataFrame,
    feature_cols: List[str],
) -> List[Dict[str, Any]]:
    """识别高相关特征对（冗余诊断）。"""
    valid = df_features[feature_cols].dropna()
    if len(valid) < MIN_SAMPLES:
        return []

    corr = valid.corr(method="spearman").abs()
    pairs: List[Dict[str, Any]] = []
    seen = set()
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            c = corr.iloc[i, j]
            if pd.isna(c):
                continue
            if c >= REDUNDANCY_CORR_THRESHOLD:
                a, b = feature_cols[i], feature_cols[j]
                key = frozenset({a, b})
                if key in seen:
                    continue
                seen.add(key)
                pairs.append({"a": a, "b": b, "corr": float(c)})
    pairs.sort(key=lambda p: -p["corr"])
    return pairs


def build_report(
    price_frames: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    horizons: Sequence[int] = (5, 10, 20),
    transform_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """全池特征信息量审计报告。

    Parameters
    ----------
    price_frames : {symbol: DataFrame} 原始行情
    feature_cols : 特征列名列表
    horizons : 预测周期列表
    transform_fn : callable(df, horizon) -> df_features，用于计算特征
    """
    report: Dict[str, Any] = {
        "command": "feature-informativeness",
        "kind": "feature_informativeness_audit",
        "horizons": list(horizons),
        "symbols": list(price_frames.keys()),
        "n_features": len(feature_cols),
        "per_horizon": {},
        "redundant_pairs": [],
        "zero_info_features": [],
        "affects_gate": False,
    }

    all_frames: List[pd.DataFrame] = []

    for h in horizons:
        per_symbol: Dict[str, Any] = {}
        pooled_mi: Dict[str, List[float]] = {c: [] for c in feature_cols}
        pooled_ic: Dict[str, List[float]] = {c: [] for c in feature_cols}

        for sym, df in price_frames.items():
            if transform_fn is not None:
                try:
                    df_feat = transform_fn(df, h)
                except Exception as e:  # noqa: BLE001
                    per_symbol[sym] = {"available": False, "reason": str(e)}
                    continue
            else:
                df_feat = df

            result = evaluate_symbol(df_feat, feature_cols, h)
            per_symbol[sym] = result

            if result.get("available"):
                for c in feature_cols:
                    pooled_mi[c].append(result["mi"].get(c, 0.0))
                    pooled_ic[c].append(result["ic"].get(c, 0.0))
                all_frames.append(df_feat)

        mi_mean = {c: float(np.mean(v)) if v else 0.0 for c, v in pooled_mi.items()}
        ic_mean = {c: float(np.mean(v)) if v else 0.0 for c, v in pooled_ic.items()}
        ic_abs_mean = {c: float(np.mean(np.abs(v))) if v else 0.0 for c, v in pooled_ic.items()}

        mi_rank = sorted(feature_cols, key=lambda c: -mi_mean.get(c, 0.0))
        ic_rank = sorted(feature_cols, key=lambda c: -ic_abs_mean.get(c, 0.0))

        zero_info = [c for c in feature_cols if mi_mean.get(c, 0.0) < MI_ZERO_THRESHOLD]

        report["per_horizon"][f"{h}d"] = {
            "per_symbol": per_symbol,
            "mi_mean": mi_mean,
            "ic_mean": ic_mean,
            "ic_abs_mean": ic_abs_mean,
            "mi_rank_top20": mi_rank[:20],
            "mi_rank_bottom10": mi_rank[-10:],
            "ic_rank_top20": ic_rank[:20],
            "ic_rank_bottom10": ic_rank[-10:],
            "zero_info_features": zero_info,
            "n_zero_info": len(zero_info),
        }

    if all_frames:
        big = pd.concat(all_frames, ignore_index=True)
        report["redundant_pairs"] = compute_redundancy(big, feature_cols)

    all_zero = set(feature_cols)
    for h in horizons:
        hk = f"{h}d"
        if hk in report["per_horizon"]:
            all_zero &= set(report["per_horizon"][hk]["zero_info_features"])
    report["zero_info_features"] = sorted(all_zero)

    return report