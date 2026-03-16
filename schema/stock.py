# Định nghĩa schema tối ưu bộ nhớ cho dữ liệu chứng khoán
STOCK_SCHEMA = {
    "symbol": "category",
    "date": "datetime64[ns]",
    "open": "float32",
    "high": "float32",
    "low": "float32",
    "close": "float32",
    "volume": "int64",
    "adj_close": "float32"
}
