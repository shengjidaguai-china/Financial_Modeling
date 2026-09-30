"""FinancialData.Net 期权数据客户端（fdnpy 封装；美股期权链 / 价格 / Greeks）。

定位（与 akshare_client 互补）：
  akshare 覆盖 美股日K + 海外期货 + 美股基本面（免费），但**无美股期权**；
  fdnpy 补这一块——option-chain / option-prices / option-greeks（Standard 订阅）。

设计约束（与 akshare_client 同构，fail-open）：
- **绝不向上抛异常**：未安装 fdnpy / 未配 FDN_API_KEY / 网络 / 解析一律返回 None；
- 未安装 fdnpy 时**静默跳过**（`ImportError`），CI 无网环境不受影响；
- 缺 API Key 时**静默跳过**并记一次 warning（不刷屏：用 `_key_warned` 标志位）；
- 单测全部走注入式 `_client` mock，**不触网**；
- 期权是**非 OHLCV 资产类别**，不进 collector 日K优先级链，由调用方按需独立取数。

环境变量：
  FDN_API_KEY — FinancialData.Net API Key（Standard 订阅档起支持期权）。
  缺省不安装 fdnpy（见 requirements.txt 可选依赖注释）；缺失时本模块全部降级返回 None。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# 环境变量名（与项目 WIND_API_KEY 同风格）
FDN_API_KEY_ENV = "FDN_API_KEY"


class FdnClient:
    """FinancialData.Net 期权客户端（fdnpy；美股期权链 / 价格 / Greeks）。

    fail-open：未安装 / 缺 Key / 接口异常一律返回 None，不阻断主链路。
    调用方注入 `config` 以读取超时；所有失败降级而非抛异常。
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        data_cfg = self.config.get("data", {}) or {}
        self.timeout = float(data_cfg.get("fdn_timeout", 30))
        # API Key：优先环境变量，其次 config.data.fdn_api_key（勿写入版本库）
        self._api_key = os.environ.get(FDN_API_KEY_ENV) or str(
            data_cfg.get("fdn_api_key", "") or ""
        )
        # 懒加载 fdnpy 客户端
        self._client: Any = None
        # 缺 Key 时只 warning 一次（避免批量调用刷屏）
        self._key_warned = False

    # ------------------------------------------------------------------
    def available(self) -> bool:
        """是否可用（fdnpy 已安装 且 API Key 已配）。供调用方预检。"""
        return self._get_client() is not None

    def _get_client(self) -> Any:
        """懒加载 fdnpy FinancialDataClient；未安装 / 缺 Key 返回 None（fail-soft）。"""
        if self._client is not None:
            return self._client
        if not self._api_key:
            if not self._key_warned:
                logger.warning(
                    "[fdn] 未配置 %s，期权数据源跳过"
                    "（Standard 订阅档 Key，见 https://financialdata.net/pricing）",
                    FDN_API_KEY_ENV,
                )
                self._key_warned = True
            return None
        try:
            from fdnpy import FinancialDataClient  # type: ignore
        except ImportError:
            logger.debug("[fdn] 未安装 fdnpy，跳过（pip install fdnpy 可启用）")
            return None
        try:
            self._client = FinancialDataClient(api_key=self._api_key)
        except Exception as e:  # noqa: BLE001  fail-soft
            logger.warning(f"[fdn] fdnpy 客户端初始化失败: {e}")
            return None
        return self._client

    # ------------------------------------------------------------------
    def fetch_option_chain(self, symbol: str) -> Optional[pd.DataFrame]:
        """美股期权链（contract_name / expiration_date / put_or_call / strike_price）。

        symbol 为美股代码（如 'MSFT'）；返回全部在市合约清单。失败返回 None。
        """
        client = self._get_client()
        if client is None:
            return None
        try:
            rows = client.get_option_chain(identifier=symbol)
        except Exception as e:  # noqa: BLE001  fail-open
            logger.warning(f"[fdn] {symbol} 期权链拉取失败: {e}")
            return None
        return self._to_frame(rows, symbol, "option_chain")

    def fetch_option_prices(self, contract: str) -> Optional[pd.DataFrame]:
        """单个期权合约的日K（date / open / high / low / close / volume）。

        contract 为完整合约名（如 'MSFT260123C00455000'）。失败返回 None。
        """
        client = self._get_client()
        if client is None:
            return None
        try:
            rows = client.get_option_prices(identifier=contract)
        except Exception as e:  # noqa: BLE001  fail-open
            logger.warning(f"[fdn] {contract} 期权价格拉取失败: {e}")
            return None
        return self._to_frame(rows, contract, "option_prices")

    def fetch_option_greeks(self, contract: str) -> Optional[pd.DataFrame]:
        """单个期权合约的 Greeks（date / delta / gamma / theta / vega / rho）。

        contract 为完整合约名。失败返回 None。
        """
        client = self._get_client()
        if client is None:
            return None
        try:
            rows = client.get_option_greeks(identifier=contract)
        except Exception as e:  # noqa: BLE001  fail-open
            logger.warning(f"[fdn] {contract} Greeks 拉取失败: {e}")
            return None
        return self._to_frame(rows, contract, "option_greeks")

    # ------------------------------------------------------------------
    @staticmethod
    def _to_frame(
        rows: Any, identifier: str, kind: str
    ) -> Optional[pd.DataFrame]:
        """fdnpy 返回 list[dict] → DataFrame；空 / 非法一律返回 None。"""
        if not rows:
            logger.warning(f"[fdn] {identifier}({kind}) 返回空")
            return None
        try:
            df = pd.DataFrame(rows)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[fdn] {identifier}({kind}) 转 DataFrame 失败: {e}")
            return None
        if df.empty:
            return None
        return df