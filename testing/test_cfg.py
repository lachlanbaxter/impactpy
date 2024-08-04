import polars as pl
from pathlib import Path

import sys
sys.path.append('../')
from data.processing.data_processing import *

### Global vars ###

DATA_DIR = Path().resolve().parents[0] / 'data' / 'raw_bin_samples'
FILES: list[Path] = list(DATA_DIR.iterdir())
DTYPES = {'date': pl.Date, 'time': pl.String, 'stock': pl.Categorical,
	'midEnd': pl.Float64, 'trade': pl.Int32}
TO_REMOVE = {'stock': ['APC']}
WINDOW = 30

### Compound functions ###

def get_all_data(files: list[Path]) -> pl.DataFrame:
	lf = load_raw_csvs(files, DTYPES)
	operations = {
		# Remove APC since we have incomplete information
		remove_rows: {'to_remove': TO_REMOVE},
		# Add rows for time bins that we have information for
		add_missing_rows: {},
		# Forward then backward fill prices and fill trade rows with 0
		fill_lf: {},
		# Absolute value of trade col
		add_volume_col: {}
	}
	for operation, kwargs in operations.items():
		lf = operation(lf, **kwargs)
	with pl.StringCache():
		df = lf.collect()
	return df

def get_dfs(files: list[Path] = FILES) -> tuple[pl.DataFrame, pl.DataFrame]:
	all_df = get_all_data(files)
	daily_df = get_daily_stats(all_df, window=WINDOW).collect()
	return all_df, daily_df