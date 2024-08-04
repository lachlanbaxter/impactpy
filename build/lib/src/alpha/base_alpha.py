from abc import ABC
from schedule import Schedule

class BaseAlpha(ABC):

	@property
	def schedule(self):
		return self._schedule

	def __init__(self, schedule: Schedule):
		self._schedule = schedule