from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import (
	Iterable, Optional, Mapping, Any, NamedTuple, Collection, final
)
from datetime import datetime, timedelta, time, date
from enum import Enum
import pandas as pd
import polars as pl


### Types


type Data = pd.DataFrame | pl.DataFrame | pl.LazyFrame # Distinguish bin/event data


def event_to_binned(data: Data, **params: dict[str, Any]) -> Data: ...


### Impact Models


class MarketParams(NamedTuple):
	"""Parameters governing trading in a market."""

	open: time
	close: time
	tickers: Collection[str]
	trading_days: Iterable[date]
	latency: timedelta


@dataclass
class Model:

	is_universal: bool
	rolling_window: int

	@abstractmethod
	def advance_time(self, dt: timedelta) -> None:
		"""Evolve model to reflect a period of time passing."""

	@abstractmethod
	def jump_to_time(self, target_time: datetime) -> None:
		"""Evolve model to reflect time passing until a given point."""

	@abstractmethod
	def jump_to_eod(self) -> None:
		"""Evolve model to reflect time passing until market close."""

	@abstractmethod
	def quote(self, order: Order) -> float:
		"""Price an order. Do not execute it."""

	@abstractmethod
	def execute(self, order: Order) -> float:
		"""Execute an order. Return cost."""

	@abstractmethod
	def fit(self, data: Data) -> None:
		"""Fit model to market data."""

@dataclass
class ImpactModel(Model):

	half_life: float
	temporary_impact_state: Mapping[str, float]
	permanent_impact_state: Mapping[str, float]
	fundamental_price: Mapping[str, float]

class OWModel(ImpactModel): ...
class RFModel(ImpactModel): ...
class AFSModel(ImpactModel): ...

class LOBModel(Model): ...


### Signals


class Alpha: ...


### Books


class Book(ABC):
	"""Represents quoted prices of a collection of tickers."""

	@property
	def curr_time(self) -> datetime:
		return self._curr_time
	
	@curr_time.setter
	def curr_time(self, new_time: datetime) -> None:
		"""Set market time."""
		# TODO: ERROR CHECK
		self._curr_time = new_time

	@abstractmethod
	def advance_time(self, dt: timedelta) -> None:
		"""Evolve market to reflect a period of time passing."""

	@abstractmethod
	def jump_to_time(self, target_time: datetime) -> None:
		"""Evolve market to reflect time passing until a given point."""

	@abstractmethod
	def jump_to_eod(self) -> None:
		"""Evolve market to reflect time passing until market close."""

	@abstractmethod
	def quote(self, order: Order) -> float:
		"""Price an order. Do not execute it."""

	@abstractmethod
	def execute(self, order: Order) -> float:
		"""Execute an order. Return cost."""


class ModeledBook(Book):

	__slots__ = ['_model']

	@property
	def model(self) -> ImpactModel:
		return self._model
	
	@abstractmethod
	def __init__(self, model: ImpactModel): ...
	
	@final
	def advance_time(self, dt: timedelta) -> None:
		self.model.advance_time(dt)

	@final
	def jump_to_time(self, target_time: datetime) -> None:
		self.model.jump_to_time(target_time)

	@final
	def jump_to_eod(self) -> None:
		self.model.jump_to_eod()
	
	def quote(self, order: Order) -> float:
		return self.model.quote(order)

	def execute(self, order: Order) -> float:
		return self.model.execute(order)


class MidOnlyBook(ModeledBook):
	"""All quotes assume initially bid = ask = mid."""

	def __init__(self, model: MidImpactModel) -> None:
		self._model = model

	def quote(self, order: Order) -> float:
		if isinstance(order, LimitOrder):
			raise InvalidOrderError(f'Cannot quote limit order on {self.__class__.__name__}')
		return super().quote(order)
	
	def execute(self, order: Order) -> float:
		if isinstance(order, LimitOrder):
			raise InvalidOrderError(f'Cannot execute limit order on {self.__class__.__name__}')
		return super().execute(order)


class ContinuousLOB(ModeledBook):
	"""Assume functional form of both sides of the LOB."""

	def __init__(self, model: LOBImpactModel) -> None:
		self._model = model


class DiscreteLOB(Book): ...


### Schedules


class Schedule(ABC): ...
class VWAP(Schedule): ...
class TWAP(Schedule): ...


### Algo


@dataclass
class Algo:

	impact_model: ImpactModel
	alphas: Iterable[Alpha]
	orders: Iterable[Order]
	book: Book
	time: datetime
	alternate_scheduler: Optional[Schedule]

	def add_alpha(self, alpha: Alpha) -> None: ...

	def add_order(self, order: Order) -> None: ...

	def get_trades(self) -> Order:
		"""Use some sort of generator to yield trades. Can be c++ co-routine."""