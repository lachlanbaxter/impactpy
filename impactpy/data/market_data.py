from dataclasses import dataclass
import polars as pl
from .bin_schedule import BinSchedule

@dataclass
class ColumnBindings:

	symbol: str
	date: str
	time: str
	price: str
	trade: str

	def __call__(self, df: pl.LazyFrame | pl.DataFrame) -> None:
		expected_types = {
			self.symbol: pl.Categorical,
			self.date: pl.Date,
			self.time: pl.Time,
			self.price: pl.Float64,
			self.trade: pl.Int32
		}
		schema = df.schema
		for col, type_ in expected_types.items():
			if schema[col] != type_:
				raise TypeError(f'Col "{col}" is "{schema[col]}", expected "{type_}"')


@dataclass
class BinnedData:

	data: pl.LazyFrame
	schedule: BinSchedule
	bindings: ColumnBindings

	def __post_init__(self):
		self.bindings(self.data)
		idx = self.schedule()
		self.data = idx.join(self.data, on=['symbol', 'date', 'time'], how='left')

	