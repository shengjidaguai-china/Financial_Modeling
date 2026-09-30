"""fdnpy 期权数据源测试：缺 Key 降级 / 未安装降级 / 期权链·价格·Greeks / fail-open。

测试要点（对照 akshare_client 同构原则）：
  - 缺 FDN_API_KEY（环境变量与 config 均无）→ available() False，全部 fetch 返回 None；
  - 缺 Key 只 warning 一次（批量调用不刷屏）；
  - 未安装 fdnpy（ImportError）→ 静默返回 None，不抛异常；
  - 有 Key + fdnpy 可用 → 期权链/价格/Greeks 正确转 DataFrame；
  - fdnpy 抛异常 / 空返回 → None（fail-open）；
  - Key 优先级：环境变量 > config.data.fdn_api_key。

全部离线（fdnpy 调用一律注入 fake，不触网）。
"""
from __future__ import annotations

import sys
from pathlib import Path


import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fdn_client import FDN_API_KEY_ENV, FdnClient  # noqa: E402


class _FakeFdnClient:
    """假 fdnpy FinancialDataClient：记录调用并返回预设结果/异常。"""

    def __init__(self, mapping=None):
        self.api_key = ""
        self.calls = []
        self.mapping = mapping or {}

    def _make(self, name):
        def _fn(**kwargs):
            self.calls.append((name, kwargs))
            value = self.mapping.get(name)
            if isinstance(value, Exception):
                raise value
            return value
        return _fn

    def __getattr__(self, name):
        return self._make(name)


class _FakeFdnModule:
    """假 fdnpy 模块：FinancialDataClient 工厂返回固定实例。"""

    def __init__(self, instance):
        self._instance = instance

    def FinancialDataClient(self, api_key=""):
        self._instance.api_key = api_key
        return self._instance


def _install_fdn(monkeypatch, instance):
    monkeypatch.setitem(sys.modules, "fdnpy", _FakeFdnModule(instance))


# ----------------------------------------------------------------------
# 缺 Key / 未安装降级
# ----------------------------------------------------------------------
class TestDegradation:
    def test_no_key_returns_none(self, monkeypatch):
        monkeypatch.delenv(FDN_API_KEY_ENV, raising=False)
        c = FdnClient({"data": {}})
        assert c.available() is False
        assert c.fetch_option_chain("MSFT") is None
        assert c.fetch_option_prices("MSFT260123C00455000") is None
        assert c.fetch_option_greeks("MSFT260123C00455000") is None

    def test_no_key_warned_once(self, monkeypatch, caplog):
        monkeypatch.delenv(FDN_API_KEY_ENV, raising=False)
        c = FdnClient({"data": {}})
        c.fetch_option_chain("MSFT")
        c.fetch_option_chain("AAPL")
        warns = [r for r in caplog.records if "未配置" in r.message]
        assert len(warns) == 1, "缺 Key 只 warning 一次，不刷屏"

    def test_not_installed_returns_none(self, monkeypatch):
        monkeypatch.setenv(FDN_API_KEY_ENV, "fake_key")
        monkeypatch.delitem(sys.modules, "fdnpy", raising=False)
        import builtins

        real_import = builtins.__import__

        def _boom(name, *a, **k):
            if name == "fdnpy":
                raise ImportError("not installed")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _boom)
        c = FdnClient({"data": {}})
        assert c.available() is False
        assert c.fetch_option_chain("MSFT") is None

    def test_key_from_config(self, monkeypatch):
        monkeypatch.delenv(FDN_API_KEY_ENV, raising=False)
        c = FdnClient({"data": {"fdn_api_key": "cfg_key"}})
        assert c._api_key == "cfg_key"
        assert c.available() is False  # 有 key 但 fdnpy 未装仍 None

    def test_env_key_preferred_over_config(self, monkeypatch):
        monkeypatch.setenv(FDN_API_KEY_ENV, "env_key")
        c = FdnClient({"data": {"fdn_api_key": "cfg_key"}})
        assert c._api_key == "env_key"


# ----------------------------------------------------------------------
# 期权链 / 价格 / Greeks（注入 fake fdnpy）
# ----------------------------------------------------------------------
class TestOptionFetch:
    def test_option_chain_success(self, monkeypatch):
        monkeypatch.setenv(FDN_API_KEY_ENV, "fake_key")
        fake = _FakeFdnClient(mapping={
            "get_option_chain": [
                {"trading_symbol": "MSFT", "contract_name": "MSFT271217P00660000",
                 "expiration_date": "2027-12-17", "put_or_call": "Put", "strike_price": 660.0},
                {"trading_symbol": "MSFT", "contract_name": "MSFT271217C00660000",
                 "expiration_date": "2027-12-17", "put_or_call": "Call", "strike_price": 660.0},
            ],
        })
        _install_fdn(monkeypatch, fake)
        c = FdnClient({"data": {}})
        df = c.fetch_option_chain("MSFT")
        assert df is not None and len(df) == 2
        assert fake.calls[0][0] == "get_option_chain"
        assert fake.calls[0][1]["identifier"] == "MSFT"
        assert "contract_name" in df.columns

    def test_option_prices_success(self, monkeypatch):
        monkeypatch.setenv(FDN_API_KEY_ENV, "fake_key")
        fake = _FakeFdnClient(mapping={
            "get_option_prices": [
                {"contract_name": "MSFT250417C00400000", "date": "2025-03-07",
                 "open": 11.45, "high": 11.9, "low": 8.75, "close": 11.25, "volume": 1005.0},
            ],
        })
        _install_fdn(monkeypatch, fake)
        c = FdnClient({"data": {}})
        df = c.fetch_option_prices("MSFT250417C00400000")
        assert df is not None and len(df) == 1
        assert df.iloc[0]["close"] == pytest.approx(11.25)

    def test_option_greeks_success(self, monkeypatch):
        monkeypatch.setenv(FDN_API_KEY_ENV, "fake_key")
        fake = _FakeFdnClient(mapping={
            "get_option_greeks": [
                {"contract_name": "X", "date": "2025-03-07", "delta": 0.16,
                 "gamma": 0.0001, "theta": -0.02, "vega": 0.32, "rho": 0.07},
            ],
        })
        _install_fdn(monkeypatch, fake)
        c = FdnClient({"data": {}})
        df = c.fetch_option_greeks("X")
        assert df is not None and len(df) == 1
        assert "delta" in df.columns

    def test_api_exception_returns_none(self, monkeypatch):
        monkeypatch.setenv(FDN_API_KEY_ENV, "fake_key")
        fake = _FakeFdnClient(mapping={"get_option_chain": RuntimeError("boom")})
        _install_fdn(monkeypatch, fake)
        c = FdnClient({"data": {}})
        assert c.fetch_option_chain("MSFT") is None

    def test_empty_rows_returns_none(self, monkeypatch):
        monkeypatch.setenv(FDN_API_KEY_ENV, "fake_key")
        fake = _FakeFdnClient(mapping={"get_option_chain": []})
        _install_fdn(monkeypatch, fake)
        c = FdnClient({"data": {}})
        assert c.fetch_option_chain("MSFT") is None

    def test_client_init_exception_returns_none(self, monkeypatch):
        """fdnpy 装了但 FinancialDataClient(api_key=...) 抛异常 → None。"""
        monkeypatch.setenv(FDN_API_KEY_ENV, "fake_key")

        class _BoomModule:
            def FinancialDataClient(self, api_key=""):
                raise RuntimeError("init failed")

        monkeypatch.setitem(sys.modules, "fdnpy", _BoomModule())
        c = FdnClient({"data": {}})
        assert c.available() is False
        assert c.fetch_option_chain("MSFT") is None