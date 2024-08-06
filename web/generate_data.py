import yfinance as yf
import pandas as pd

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Masukkan simbol saham yang ingin Anda analisis
ticker = 'AAPL'
start_date = '2022-01-01'
end_date = '2022-12-31'

stock_data = get_stock_data(ticker, start_date, end_date)

# Proses untuk menyiapkan data harga saham
stock_data['Date'] = stock_data.index
stock_data.reset_index(drop=True, inplace=True)
stock_data = stock_data[['Date', 'Open', 'High', 'Low', 'Close']]

# Simpan data harga saham dalam format CSV
stock_data.to_csv('stock_data.csv', index=False)