from typing import Mapping, Optional
import polars as pl
from enum import Enum


def roll_avg(lf: pl.LazyFrame, col: str, window: int) -> pl.LazyFrame:
	return (
		lf.rolling(group_by='symbol', index_column='date', period=f'{window}d')
		.agg(pl.mean(col))
	)


def daily_vola(lf: pl.LazyFrame) -> pl.LazyFrame:
	daily_vola_expr = pl.col('price').pct_change().mul(100).std()

	return (lf
		.group_by(['symbol', 'date'], maintain_order=True)
		.agg(daily_vola_expr.alias('daily_vola'))
		.select(*[pl.col(col) for col in ['symbol', 'date', 'daily_vola']])
	)


def daily_volu(lf: pl.LazyFrame) -> pl.LazyFrame:
	daily_volu_col = pl.col('trade').abs()

	return (lf
		.with_columns(daily_volu_col.alias('daily_volu'))
		.group_by(['symbol', 'date'], maintain_order=True)
		.sum()
		.select(*[pl.col(col) for col in ['symbol', 'date', 'daily_volu']])
	)


def enhance_data(df: pl.DataFrame, window: int) -> pl.DataFrame:
	lf = df.lazy()
	daily_vola_lf = daily_vola(lf)
	daily_volu_lf = daily_volu(lf)
	daily_df = daily_vola_lf.join(daily_volu_lf, on=['symbol', 'date']).collect()
	
	avg_cols = {'daily_vola': 'vola', 'daily_volu': 'adv'}
	roll_lf = roll_avg(daily_df, avg_cols, window)
	lf = lf.join(roll_lf, how='left', on=['symbol', 'date']).drop('date')
	return lf.collect()


def compute_OW_impact(
		trades: pl.DataFrame,
		push: pl.DataFrame,
		alpha: float,
) -> pl.DataFrame:

	return (trades.lazy()
		.with_columns(pl.col('trade').mul('vola').truediv('adv'))
		
		# EWM = ax_n-1 + (1-a)x_n, but we want ax_n-1 + x_n to approximate
		# exponential decay of the impact state. So divide by 1-a first.
		.with_columns(pl.lit(1 - alpha).alias('norm'))
		# Don't divide first impact state by 1-a.
		.with_columns(pl.col('norm').shift().over(['symbol', 'date']).fill_null(1))
		.with_columns(pl.col('trade').truediv('norm'))
		.drop('norm')
		
		.with_columns(pl.col('trade').ewm_mean(
			alpha=1 - alpha, adjust=False, ignore_nulls=True
		).over(['symbol', 'date']))
		# .with_columns(pl.lit(1).alias('push'))
		.join(push.lazy(), on='symbol', how='left')
		.with_columns(pl.col('trade').mul('push').alias('impact'))
	).collect()


SampleGroup = Enum('SampleGroup', ['TRAIN', 'VALIDATION', 'TEST'])


_REG_FIT_EXPRS = (
	('cov_xy', pl.col('e_xy') - pl.col('e_x') * pl.col('e_y')),
	('var_x', pl.col('e_xx') - pl.col('e_x') ** 2),

	('beta', pl.col('cov_xy') / pl.col('var_x')),
	('alpha', pl.col('e_y') - pl.col('beta') * pl.col('e_x')),
)


_REG_STAT_EXPRS = (
	('mse', pl.col('e_yy') + pl.col('beta') ** 2 * pl.col('e_xx') + pl.col('alpha') ** 2
		+ 2 * (
			pl.col('beta') * pl.col('e_x') * pl.col('alpha')
			- pl.col('e_xy') * pl.col('beta')
			- pl.col('e_y') * pl.col('alpha')
		)
	),
	('rmse', pl.col('mse').sqrt()),
	('tss', (pl.col('e_yy') - pl.col('e_y') ** 2) * pl.col('n')),
	('rss', pl.col('mse') * pl.col('n')),
	('rsq', 1 - pl.col('rss') / pl.col('tss')),
)


def simple_regression_core(lf: pl.LazyFrame) -> pl.LazyFrame:
	"""
	OLS fit of y = beta * x + alpha.
	
	Fitted to the data labelled SampleGroup.TRAIN in the `sample` column
	and then tested on the other samples.

	This is repeated for block of data with the same value in the `group` column.
	"""

	lf = (
		lf.with_columns(
			# Add products for later calculations of E[XX], E[YY], E[XY]
			xx=pl.col('x').mul('x'),
			yy=pl.col('y').mul('y'),
			xy=pl.col('x').mul('y'),
		)
		.group_by(['group', 'sample'], maintain_order=True)
		# Calculate expectations: e_xy = E[XY] = sum_{i=1}^n x_i * y_i / n
		.agg(pl.all().sum().truediv(pl.len()).name.prefix('e_'), n=pl.len())
	)

	# Calculate regression coefficients for training samples
	train_lf = lf.filter(pl.col('sample').eq(SampleGroup.TRAIN.value))
	for col, expr in _REG_FIT_EXPRS:
		train_lf = train_lf.with_columns(expr.alias(col))

	# Broadcast coefficients to out of sample data
	lf = lf.join(train_lf.select(['group', 'beta', 'alpha']), on='group',
		how='left', join_nulls=True)

	# Calculate fit statistics
	for col, expr in _REG_STAT_EXPRS:
		lf = lf.with_columns(expr.alias(col))

	return lf


def simple_regression(
		df: pl.DataFrame,
		x_col: str,
		y_col: str,
		group_col: Optional[str] = None,
		sample_col: Optional[str] = None,
) -> pl.DataFrame:
	"""Wrapper to simple_regression_core."""
	
	lf = (
		df.lazy()
		.select(
			# Rename columns and optionally fill with dummies
			x=pl.col(x_col),
			y=pl.col(y_col),
			group=pl.col(group_col) if group_col else 1,
			# TODO: Coerce into Enum, maybe using pl.Enum
			sample=pl.col(sample_col) if sample_col else pl.lit(SampleGroup.TRAIN.value),
		)
		.drop_nulls()
	)

	lf = simple_regression_core(lf)

	if not group_col: lf = lf.drop('group')
	if not sample_col: lf = lf.drop('sample')

	return lf.collect()


def fit_OW_regression(
		df: pl.DataFrame,
		push: pl.DataFrame,
		half_life: float,
		window: int,
		cols: Mapping[str, str],
		is_universal: bool,
) -> pl.DataFrame:
	
	df = df.select(
		# TODO Should include pl.col here?
		timestamp=pl.col(cols['timestamp']),
		date=pl.col(cols['timestamp']).dt.date(),
		symbol=pl.col(cols['symbol']),
		price=pl.col(cols['price']),
		trade=pl.col(cols['trade']),
	)

	df = enhance_data(df, window)

	# Add date col!

	return df