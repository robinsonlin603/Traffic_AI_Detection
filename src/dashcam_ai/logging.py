"""設定並取得專案共用的結構化日誌。"""

import logging
from typing import Any

import structlog


def configure_logging(level: str = "INFO") -> None:
    """設定標準日誌與 structlog，讓每筆訊息以 JSON 格式輸出。"""
    logging.basicConfig(format="%(message)s", level=level.upper())
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
    )


def get_logger() -> Any:
    """取得已套用專案設定的結構化 logger。"""
    return structlog.get_logger()
