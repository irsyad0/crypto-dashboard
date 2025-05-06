# crypto_dashboard.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
from PIL import Image
from sklearn.linear_model import LinearRegression
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA

# --- PAGE CONFIG ---
st.set_page_config(page_title="Crypto Dashboard", layout="wide")

# --- HEADER ---
try:
    custom_image = Image.open("gambar.png")
    st.image(custom_image, use_container_width=True)
except Exception as e:
    st.warning(f"Gagal memuat custom image: {e}")

st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔍 Pengaturan")
    crypto_dict = {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "BNB": "BNB-USD",
        "Cardano": "ADA-USD",
        "Solana": "SOL-USD",
        "XRP": "XRP-USD",
        "Dogecoin": "DOGE-USD",
        "Polkadot": "DOT-USD",
        "Litecoin": "LTC-USD",
        "Chainlink": "LINK-USD",
        "Tether": "USDT-USD",
        "Pepe": "PEPE24478-USD",
        "Shiba Inu": "SHIB-USD"
    }
    pilihan = st.multiselect("Pilih crypto:", list(crypto_dict.keys()), default=["Bitcoin", "Ethereum"])
    jumlah_hari = st.slider("Prediksi berapa hari ke depan?", 1, 30, 7)
    model_choice = st.radio("Pilih model prediksi:", ["ARIMA", "Linear Regression"])

if not pilihan:
    st.warning("Silakan pilih minimal satu crypto untuk ditampilkan.")
    st.stop()

tickers = [crypto_dict[n] for n in pilihan]

# --- DATA ---
@st.cache_data

def ambil_data(tickers):
    start_date = "2015-01-01"
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]

    for col in data.columns:
        first_valid = data[col].first_valid_index()
        if first_valid:
            data.loc[:first_valid, col] = 0
        else:
            data[col] = 0

    return data.fillna(0)

with st.spinner("Mengambil data dari Yahoo Finance..."):
    data_harga = ambil_data(tickers)

if data_harga.empty:
    st.warning("Data tidak tersedia untuk crypto yang dipilih.")
    st.stop()

st.markdown(f"📅 Data historis dari **{data_harga.index.min().date()}** sampai **{data_harga.index.max().date()}**")

# --- MODEL PREDIKSI ---
prediksi_df = pd.DataFrame()
progress = st.progress(0)

for idx, t in enumerate(tickers):
    df_ts = (
        data_harga[t]
        .reset_index()
        .rename(columns={t: "Close"})
        .set_index('Date')
        .asfreq('D')
        .ffill()
        .bfill()
        .dropna()
    )

    if df_ts.isnull().values.any() or len(df_ts) < 30:
        st.warning(f"Data untuk {t} tidak cukup untuk prediksi, dilewati.")
        continue

    try:
        if model_choice == "ARIMA":
            model_full = ARIMA(df_ts['Close'], order=(5,1,0)).fit()
            future_pred = model_full.forecast(steps=jumlah_hari)
        else:
            df_ts = df_ts.reset_index()
            df_ts["Days"] = (df_ts["Date"] - df_ts["Date"].min()).dt.days
            X = df_ts[["Days"]]
            y = df_ts["Close"]
            model_lr = LinearRegression().fit(X, y)
            future_days = np.array([[X["Days"].max() + i] for i in range(1, jumlah_hari+1)])
            future_pred = model_lr.predict(future_days)

        future_pred = np.where(future_pred < 0, 0, future_pred)
        future_dates = pd.date_range(start=df_ts.iloc[-1]['Date'] + timedelta(days=1), periods=jumlah_hari, freq='D')
        prediksi_df[t] = pd.Series(future_pred, index=future_dates)
    except Exception as e:
        st.warning(f"Gagal membuat prediksi untuk {t}: {e}")

    progress.progress((idx + 1) / len(tickers))

# --- VISUALISASI ---
st.subheader("📈 Grafik Harga & Prediksi")

hist = data_harga.reset_index().melt(id_vars="Date", var_name="Crypto", value_name="Close")
hist["Type"] = "Historis"

pred = prediksi_df.reset_index().melt(id_vars="index", var_name="Crypto", value_name="Close").rename(columns={"index": "Date"})
pred["Type"] = "Prediksi"

all_df = pd.concat([hist, pred])

chart = alt.Chart(all_df).mark_line(size=2.5).encode(
    x="Date:T",
    y="Close:Q",
    color=alt.Color("Crypto:N", legend=alt.Legend(title="Crypto")),
    strokeDash=alt.StrokeDash("Type:N", legend=alt.Legend(title="Tipe")),
    tooltip=["Date:T", "Crypto:N", "Close:Q", "Type:N"]
).properties(
    width=1000,
    height=400
).interactive()

st.altair_chart(chart, use_container_width=True)

# --- UNDUH DATA ---
st.download_button("💾 Download Data Prediksi", prediksi_df.to_csv().encode("utf-8"), file_name="prediksi.csv")

# --- NOTIFIKASI PENURUNAN HARGA ---
st.subheader("🚨 Notifikasi Penurunan Tajam (7 Hari Terakhir)")
drop_alerts = []
for t in tickers:
    if len(data_harga[t]) >= 8:
        harga_terakhir = data_harga[t].iloc[-1]
        harga_7_hari_lalu = data_harga[t].iloc[-8]
        persentase = ((harga_terakhir - harga_7_hari_lalu) / harga_7_hari_lalu) * 100
        if persentase < -10:
            drop_alerts.append((t, persentase))

if drop_alerts:
    for t, p in drop_alerts:
        st.error(f"⚠️ {t} turun {p:.2f}% dalam 7 hari terakhir!")
else:
    st.success("✅ Tidak ada crypto yang turun lebih dari 10% dalam 7 hari terakhir.")

# --- VOLATILITAS ---
st.subheader("🔥 Volatilitas 7 Hari Terakhir")
volatility = data_harga.pct_change().rolling(window=7).std().iloc[-1] * 100
vol_df = pd.DataFrame(volatility, columns=["Volatilitas (%)"]).sort_values(by="Volatilitas (%)", ascending=False)
st.dataframe(vol_df.style.background_gradient(cmap='Reds'))

# --- METRIK ---
st.subheader("📊 Harga Terakhir")
last_prices = data_harga.iloc[-1]
if len(tickers) > 0:
    max_per_row = 5
    for i in range(0, len(tickers), max_per_row):
        row_tickers = tickers[i:i+max_per_row]
        cols = st.columns(len(row_tickers))
        for j, t in enumerate(row_tickers):
            with cols[j]:
                st.metric(label=t.replace("-USD", ""), value=f"${last_prices[t]:,.2f}")

# --- FOOTER ---
st.markdown("---")
st.caption("\n    🚧 Dibuat dengan Streamlit · Prediksi menggunakan ARIMA atau Linear Regression · Data dari Yahoo Finance\n")
