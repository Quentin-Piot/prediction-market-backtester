from __future__ import annotations

from datetime import datetime
from enum import StrEnum


class Venue(StrEnum):
    POLYMARKET = "polymarket"
    KALSHI = "kalshi"


class MarketStructure(StrEnum):
    CLOB = "clob"
    AMM = "amm"
    UNKNOWN = "unknown"


class TradeSide(StrEnum):
    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class ResolutionOutcome(StrEnum):
    YES = "yes"
    NO = "no"
    OTHER = "other"
    UNRESOLVED = "unresolved"


MarketId = str
OutcomeId = str
RunId = str
Timestamp = datetime
Probability = float
Price = float
Quantity = float
