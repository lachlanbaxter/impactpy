from dataclasses import dataclass
from typing import Sequence, Collection, Optional
from datetime import date, time, timedelta
import polars as pl


@dataclass(frozen=True, slots=True)
class BinSchedule:
	"""
	Contains symbols and trading days/hours of a market.

	Acts as a functor, providing symbols and timestamps for a submarket in
	selected time period.
	"""

	bin_duration: timedelta
	open_time: time
	close_time: time
	trading_days: Sequence[date]
	symbols: Collection[str]

	def __call__(
			self,
			*,
			start_time: Optional[time] = None,
			end_time: Optional[time] = None,
			start_date: Optional[date] = None,
			end_date: Optional[date] = None,
			included_symbols: Optional[Collection[str]] = None
	) -> pl.LazyFrame:
		"""
		Query the bin schedule.

		Any provided arguments are clamped to member variables.

		If any arguments are omitted, apply memeber variables as defaults.
		"""

		start_time = self.open_time if start_time is None \
			else max(self.open_time, start_time)
		
		end_time = self.close_time if end_time is None \
			else min(self.close_time, end_time)
		
		first_date = min(self.trading_days)
		start_date = first_date if start_date is None \
			else max(first_date, start_date)
		
		last_date = max(self.trading_days)
		end_date = last_date if end_date is None \
			else min(last_date, end_date)

		included_symbols = self.symbols if included_symbols is None \
			else set(included_symbols) & set(self.symbols)

		lf = (pl.LazyFrame()
			.with_columns(
				time = pl.datetime_range(start_date, end_date, self.bin_duration)
			)
			.filter(
				pl.col('time').dt.time().ge(start_time)
				& pl.col('time').dt.time().le(end_time)
				& pl.col('time').dt.date().is_in(self.trading_days)
			)
			.join(pl.LazyFrame({'symbol': included_symbols}), how='cross')
		)

		return lf
