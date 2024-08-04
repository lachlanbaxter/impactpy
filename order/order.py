from typing import Optional, NamedTuple
from datetime import datetime


class Order(NamedTuple):

	ticker: str
	size: float


class MarketOrder(Order): ...


class LimitOrder(Order):

	expiry: Optional[datetime] = None
