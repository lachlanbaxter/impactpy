from abc import ABC
from ..data import BinSchedule

class BaseAlpha(ABC):

	@property
	def schedule(self):
		return self._schedule

	def __init__(self, schedule: BinSchedule):
		self._schedule = schedule


class IntradayAlpha(BaseAlpha): ...


class OvernightAlpha(BaseAlpha): ...


class TradingAlpha(IntradayAlpha): ...