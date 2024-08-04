import polars as pl


# MarketData

C_STOCK = 'stock'
C_DATE = 'date'
C_TIME = 'time'

C_PRICE = 'midEnd'
C_TRADE = 'trade'

SCHEMA = {
	C_STOCK: pl.Categorical,
	C_DATE: pl.Date,
	C_TIME: pl.Time,
	C_PRICE: pl.Float64,
	C_TRADE: pl.Int32
}