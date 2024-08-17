import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
import polars as pl
from typing import Any, Self

from .subroutines import compute_OW_impact, simple_regression


class ImpactModelTransformer(TransformerMixin, BaseEstimator):
	"""Transforms market orders into an impact state and vice versa.

	Parameters
	----------
	half_life : float
		A parameter used to determine the rate at which the impact state decays.
			
	is_universal : bool, default=True
		A parameter used to control whether different symbols should be modelled
		separately.
			
	is_stochastic_push : bool, default=False
		A parameter used to indicate whether the push parameter is time varying.
			
	turn_of_day_effect : str, default='reset'
		A parameter used to control how the impact changes overnight.

	fit_style: str, default='regression'
		A parameters used to control the fitting procedure

	Attributes
	----------
	n_features_in_ : int
		Number of features seen during :term:`fit`.

	feature_names_in_ : ndarray of shape (`n_features_in_`,)
		Names of features seen during :term:`fit`. Defined only when `X`
		has feature names that are all strings.
	"""

	# For input validation in `_fit_context` decorator
	_parameter_constraints = {
		'half_life': [float],
		'is_universal': [bool],
		'is_stochastic_push': [bool],
		'turn_of_day_effect': [str],
		'fit_style': [str],
	}

	def __init__(
			self,
			*,
			half_life: float,
			is_universal: bool = True,
			is_stochastic_push: bool = False,
			turn_of_day_effect: str = 'reset',
			fit_style: str = 'regression',
	) -> None:

		self.half_life = half_life
		self.is_universal = is_universal
		self.is_stochastic_push = is_stochastic_push
		self.turn_of_day_effect = turn_of_day_effect
		self.fit_style = fit_style

	@_fit_context(prefer_skip_nested_validation=True)
	def fit(
			self,
			X: pl.LazyFrame,
			y: Any = None,
			*,
			timestamp_col: str = 'timestamp',
			symbol_col: str = 'symbol',
			price_col: str = 'price',
			trade_col: str = 'trade',
	) -> Self:
		"""Fit impact model to 

		Parameters
		----------
		X : pl.DataFrame
		  DataFrame with at least 4 columns containing timestamps,
			symbols, terminal prices, and trade sizes.

		y : None
			Ignored.

		timestamp_col : str
			Name of the column containing timestamps.

		symbol_col : str
			Name of the column containing symbols.

		price_col : str
			Name of the column containing (terminal) prices.

		trade_col : str
			Name of the column containing trade sizes.

		Returns
		-------
		self : object
			Returns self.
		"""
		X = self._validate_data(X, accept_sparse=True)

		cols = {
			'timestamp': timestamp_col,
			'symbol': symbol_col,
			'price': price_col,
			'trade': trade_col,
		}

		if self.fit_style == 'regression':
			pass
			# self.lam_ = fit_OW_regression(X, timestamp_col, symbol_col, price_col, trade_col)
		elif self.fit_style == 'optimization':
			raise NotImplementedError
		else:
			raise ValueError(f'Unrecognised fit style: {self.fit_style}')

		# Return the transformer
		return self

	def transform(self, X):
		"""A reference implementation of a transform function.

		Parameters
		----------
		X : {array-like, sparse-matrix}, shape (n_samples, n_features)
			The input samples.

		Returns
		-------
		X_transformed : array, shape (n_samples, n_features)
			The array containing the element-wise square roots of the values
			in ``X``.
		"""
		# Since this is a stateless transformer, we should not call `check_is_fitted`.
		# Common test will check for this particularly.

		# Input validation
		# We need to set reset=False because we don't want to overwrite `n_features_in_`
		# `feature_names_in_` but only check that the shape is consistent.
		X = self._validate_data(X, accept_sparse=True, reset=False)
		return np.sqrt(X)
	
	def inverse_transform(self, X): ...

	def _more_tags(self): ...
