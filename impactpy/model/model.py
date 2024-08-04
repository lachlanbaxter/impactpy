import polars as pl
from abc import ABC, abstractmethod
from typing import Any

# Maybe add assumed columns to type hints using schemas?
# e.g aggregrate_impact assumes '_trade' exists
# all assume stock, date, and time

# TODO maybe this can be added to PivotTable? or a LazyFrame utils file? 
def prepend_constant(expr: pl.Expr, value: Any, n: int) -> pl.Expr:
	expr = (expr.gather_every(1, n).extend_constant(None, n)
		.shift(n, fill_value=value))
	return expr


class Model(ABC): ...


class LOBModel(Model):
	"""To be continued..."""


class ImpactModel(Model):
	"""Models the impact state of the midprice."""
	
	def __init__(self, beta: float, lam: float) -> None:
		# TODO validate inputs nicely
		# TODO expand to handle non-universal models
		self.beta = beta
		self.lam = lam

	@abstractmethod
	def kernel(self, expr: pl.Expr) -> pl.Expr: ...

	# TODO generalise this and add to PivotTable class
	# TODO don't forget non-universal models!
	def ew_impact(self, lf: pl.LazyFrame) -> pl.LazyFrame:
		lf = lf.with_columns((pl.col('_trade')
			.gather_every(1, 1)
			.extend_constant(None, 1)
			.shift(1, fill_value=pl.col('_trade').first().mul(1 - self.beta))
			/ (self.beta))
			.over(['stock', 'date']))
		lf = lf.with_columns(pl.col('_trade').ewm_mean(alpha=1 - self.beta,
			adjust=False, ignore_nulls=False).over(['stock', 'date']))
		return lf

	def aggregate_impact(self, lf: pl.LazyFrame) -> pl.LazyFrame:
		lf = self.ew_impact(lf)
		lf = lf.with_columns((pl.col('_trade').mul(self.lam)).alias('impact'))
		lf = lf.select(pl.exclude('_trade'))
		return lf

	@abstractmethod
	def compute_impact(self, df: pl.DataFrame) -> pl.LazyFrame: ...

	def add_impact(self, df: pl.DataFrame) -> pl.LazyFrame:
		impact_lf = self.compute_impact(df)
		impacted_price_lf = impact_lf.with_columns(
			pl.col('midEnd').add(pl.col('impact')).alias('impacted_price'))
		return impacted_price_lf

	def subtract_impact(self, df: pl.DataFrame) -> pl.LazyFrame:
		inv_trade_df = df.with_columns(pl.col('trade').mul(-1))
		impacted_price_lf = self.add_impact(inv_trade_df)
		return impacted_price_lf

class OWModel(ImpactModel):

	def kernel(self, expr: pl.Expr) -> pl.Expr:
		return expr

	# TODO maybe this doesn't need to be abstract?
	def compute_impact(self, df: pl.DataFrame) -> pl.LazyFrame:
		lf = df.lazy()
		lf = lf.with_columns((pl.col('trade') / pl.col('adv')).alias('_trade'))
		lf = lf.with_columns(self.kernel(pl.col('_trade')).alias('_trade'))
		lf = lf.with_columns((pl.col('_trade') * pl.col('vol')).alias('_trade'))
		lf = self.aggregate_impact(lf)
		return lf


class AFSModel(ImpactModel): ...


class RFModel(ImpactModel): ...
