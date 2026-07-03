from __future__ import annotations

"""Kiwoom bridge client.

This module intentionally exposes data-read methods only. It does not contain
order, market-order, condition-order, or automated trading APIs.
"""

import os
import time
from typing import Any, Protocol

import requests


class KiwoomConnectionError(RuntimeError):
    """Raised when the Kiwoom data bridge is unavailable."""


class KiwoomClientProtocol(Protocol):
    def get_quote(self, code: str) -> dict[str, Any]: ...

    def get_ticks(self, code: str, limit: int = 600) -> list[dict[str, Any]]: ...

    def get_minute_candles(self, code: str, interval: int = 1, limit: int = 240) -> list[dict[str, Any]]: ...

    def get_daily_candles(self, code: str, limit: int = 400) -> list[dict[str, Any]]: ...


class KiwoomBridgeClient:
    """HTTP bridge adapter for a local Kiwoom OpenAPI process.

    Many Kiwoom deployments expose the COM/OpenAPI process through a small local
    bridge. The endpoint is configured with ``KIWOOM_BRIDGE_URL``. When it is not
    set, the provider fails closed so analysis can write a QA failure instead of
    silently falling back to uncertain data.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 5.0):
        self.base_url = (base_url or os.getenv("KIWOOM_BRIDGE_URL") or "http://127.0.0.1:8765").rstrip("/")
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "KiwoomBridgeClient":
        return cls()

    @property
    def available(self) -> bool:
        return bool(self.base_url)

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        if not self.available:
            raise KiwoomConnectionError("KIWOOM_BRIDGE_URL is not configured")
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = requests.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict) and payload.get("error"):
                    error_text = str(payload["error"])
                    if _retryable_kiwoom_error(error_text) and attempt < 2:
                        time.sleep(1.0 + attempt)
                        continue
                    raise KiwoomConnectionError(error_text)
                return payload
            except requests.HTTPError as exc:
                status_code = getattr(exc.response, "status_code", None) or getattr(response, "status_code", None)
                if status_code == 404:
                    raise KiwoomConnectionError(f"키움 브릿지 endpoint 미지원: {path}") from exc
                last_error = exc
                if attempt < 2:
                    time.sleep(1.0 + attempt)
                    continue
                raise
            except requests.RequestException as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.0 + attempt)
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise KiwoomConnectionError(f"키움 브릿지 응답 실패: {path}")

    def get_quote(self, code: str) -> dict[str, Any]:
        try:
            return dict(self._get("/quote", {"code": code}))
        except (requests.RequestException, KiwoomConnectionError):
            payload = self._get(f"/stock/{code}", {"candleDays": 80})
            if isinstance(payload, dict) and isinstance(payload.get("stock"), dict):
                stock = dict(payload["stock"])
                stock.setdefault("timestamp", payload.get("updatedAt") or stock.get("updatedAt"))
                if stock.get("tradeAmountMillion") is not None and stock.get("trade_value") is None:
                    stock["trade_value"] = _to_number(stock.get("tradeAmountMillion")) * 1_000_000
                return stock
            raise

    def get_ticks(self, code: str, limit: int = 600) -> list[dict[str, Any]]:
        payload = self._get("/ticks", {"code": code, "limit": limit})
        return list(payload if isinstance(payload, list) else payload.get("ticks", []))

    def get_minute_candles(self, code: str, interval: int = 1, limit: int = 240) -> list[dict[str, Any]]:
        payload = self._get("/candles/minute", {"code": code, "interval": interval, "limit": limit})
        return list(payload if isinstance(payload, list) else payload.get("candles", []))

    def get_daily_candles(self, code: str, limit: int = 400) -> list[dict[str, Any]]:
        try:
            payload = self._get("/candles/daily", {"code": code, "limit": limit})
        except (requests.RequestException, KiwoomConnectionError):
            try:
                payload = self._get(f"/candles/{code}", {"days": limit})
            except (requests.RequestException, KiwoomConnectionError):
                payload = self._get(f"/stock/{code}", {"candleDays": limit})
        return list(payload if isinstance(payload, list) else payload.get("candles", []))


def _to_number(value: Any) -> float:
    try:
        return float(str(value).replace(",", ""))
    except Exception:
        return 0.0


def _retryable_kiwoom_error(error_text: str) -> bool:
    text = str(error_text)
    return "CommRqData failed" in text or "result=-200" in text or "timed out" in text
