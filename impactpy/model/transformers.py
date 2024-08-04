from typing import Self
from abc import ABC, abstractmethod
from datetime import timedelta
import numpy as np

import polars as pl

from sklearn.base import (
	BaseEstimator, TransformerMixin, _fit_context # type: ignore
)

class RollingEstimateTransformer(ABC, TransformerMixin, BaseEstimator):
	"""
	Creates a feature column containing a rolling estimate calculated over the
	last `window` days.
	"""

	def __init__(self, window: int, name: str):
		self.window = window
		self.name= name

	_parameter_constraints = {'window': [int], 'name': [str]}

	def _tidy_features(
			self,
			X: pl.DataFrame,
			cols: dict[str, str]
	) -> pl.LazyFrame:
		"""
		Optionally validate features, rename columns to standard names,
		and add `date` column for convinience.
		"""

		# self._validate_data() # type: ignore
		# TODO: Add schema validation

		tidy_lf = X.lazy().rename(cols)
		tidy_lf = tidy_lf.with_columns(date=pl.col('timestamp').dt.date())
		return tidy_lf

	@_fit_context(prefer_skip_nested_validation=True) # type: ignore
	def fit(self, X: pl.DataFrame, y: None, cols: dict[str, str]) -> Self:
		tidy_lf = self._tidy_features(X, cols)

		# Polars currently doesn't support a `min_periods` argument in .rolling
		# So we cut off estimates formed from an incomplete window.
		# Timedelta should be be window - 1, but we use window because of shift.
		first_valid_date_expr = pl.col('date').min() + timedelta(days=self.window)

		self.estimates_ = (
			self._fit_core(tidy_lf)
			.rolling(group_by='symbol', index_column='date', period=f'{self.window}d')
			.agg(pl.mean(self.name))
			# Avoid look-ahead bias by shifting estimates using today's data forward
			.with_columns(pl.col(self.name).shift(n=1).over('symbol'))
			.filter(pl.col('date') >= first_valid_date_expr)
			.select(*[pl.col(c) for c in ['symbol', 'date', self.name]])
		).collect()

		return self
	
	@abstractmethod
	def _fit_core(self, lf: pl.LazyFrame) -> pl.LazyFrame:
		"""
		Create a LazyFrame containing an estimate for each symbol and date.
		Returns no other columns.
		"""

	def transform(self, X: pl.DataFrame, cols: dict[str, str]) -> pl.DataFrame:
		tidy_lf = self._tidy_features(X, cols)
		enhanced_lf = tidy_lf.join(
			self.estimates_.lazy(), how='left', on=['symbol', 'date']	
		)
		transformed_lf = self._transform_core(enhanced_lf)
		return transformed_lf.select(self.name).collect()
	
	def _transform_core(self, lf: pl.LazyFrame) -> pl.LazyFrame:
		"""Transforms estimate. Default is the identity."""
		return lf

	def fit_transform( # type: ignore
			self,
			X: pl.DataFrame,
			y: None,
			cols: dict[str, str]
	) -> pl.DataFrame:
		"""Overloaded to correctly forward metadata to fit and transform."""

		return self.fit(X, y, cols).transform(X, cols)


### Volatility Estimators


class VolatilityEstimator(RollingEstimateTransformer):

	def __init__(
			self,
			window: int,
			name: str,
			bin_duration: int,
			trading_hours: float,
	) -> None:

		super().__init__(window, name)
		self.bin_duration = bin_duration
		self.trading_hours = trading_hours

	# TODO: Maybe make this a wrapper?
	@property
	def annualization_expr(self) -> pl.Expr:
		ann_factor = np.sqrt(252 * self.trading_hours * 3600 / self.bin_duration)
		expr = pl.col(self.name).mul(ann_factor)
		return expr


class ConstantIntradayVolatilityEstimator(VolatilityEstimator):
	"""Simple mean intraday volatility."""
	
	def _fit_core(self, lf: pl.LazyFrame) -> pl.LazyFrame:
		daily_vola_expr = pl.col('price').pct_change().mul(100).std()
		# daily_vola_expr = pl.col('price').diff().std()

		return (lf
			.group_by(['symbol', 'date'], maintain_order=True)
			.agg(daily_vola_expr.alias(self.name))
			.with_columns(self.annualization_expr)
		)


# TODO: Fill this in!
class IntradayVolatilityCurveEstimator(VolatilityEstimator): ...

	# def _fit_core(self, lf: pl.LazyFrame) -> pl.LazyFrame: ...
	
	# def _transform_core(self, lf: pl.LazyFrame) -> pl.LazyFrame: ...


### Volume Estimators


class AverageDailyVolumeEstimator(RollingEstimateTransformer):
	"""Simple mean average daily volume."""

	def _fit_core(self, lf: pl.LazyFrame) -> pl.LazyFrame:
		return (lf
			.with_columns(pl.col('trade').abs().alias(self.name))
			.group_by(['symbol', 'date'], maintain_order=True)
			.sum()
		)
