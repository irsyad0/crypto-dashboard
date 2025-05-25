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
st.set_page_config(page_title="Crypto Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- HEADER (CUSTOM IMAGE REPLACES TEXT HEADER) ---
try:
    custom_image = Image.open(
        r"gambar.png"
    )
    st.image(custom_image, use_container_width=True)
except Exception as e:
    st.warning(f"Gagal memuat custom image: {e}")

st.markdown("---")

# --- SIDEBAR FILTERS ---
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
    pilihan = st.multiselect(
        "Pilih crypto:", list(crypto_dict.keys()), default=["Bitcoin", "Ethereum"]
    )
    jumlah_hari = st.slider(
        "Prediksi berapa hari ke depan?", 1, 30, 1
    )

if not pilihan:
    st.warning("Silakan pilih minimal satu crypto untuk ditampilkan.")
    st.stop()

tickers = [crypto_dict[n] for n in pilihan]

# --- DATA ---
@st.cache_data
def ambil_data(tickers):
    start_date = "2015-01-01"
    end_date = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
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


st.markdown(
    f"📅 Data historis dari **{data_harga.index.min().date()}** "
    f"sampai **{data_harga.index.max().date()}**"
)


# --- MODEL PREDIKSI ---
prediksi_df = pd.DataFrame()

for t in tickers:
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

    if df_ts.isnull().values.any():
        st.warning(f"Data untuk {t} masih mengandung NaN, dilewati.")
        continue

    try:
        model_full = ARIMA(df_ts['Close'], order=(5,1,0)).fit()
        future_pred = model_full.forecast(steps=jumlah_hari)
        future_dates = pd.date_range(start=df_ts.index[-1] + timedelta(days=1), periods=jumlah_hari, freq='D')
        future_pred = np.where(future_pred < 0, 0, future_pred)
        prediksi_df[t] = pd.Series(future_pred, index=future_dates)
    except Exception as e:
        st.warning(f"Gagal membuat prediksi untuk {t}: {e}")

# --- VISUALISASI ---
st.subheader("📈 Grafik Harga & Prediksi")

df_hist = (
    data_harga
    .reset_index()
    .melt(
        id_vars="Date", var_name="Crypto", value_name="Close"
    )
)
df_hist["Type"] = "Historis"

df_pred = (
    prediksi_df
    .reset_index()
    .melt(
        id_vars="index", var_name="Crypto", value_name="Close"
    )
    .rename(columns={"index": "Date"})
)
df_pred["Type"] = "Prediksi"

df_all = pd.concat([df_hist, df_pred])

chart = alt.Chart(df_all).mark_line(size=3).encode(
    x=alt.X("Date:T", title="Waktu"),
    y=alt.Y("Close:Q", title="Harga (USD)"),
    color=alt.Color("Crypto:N", legend=alt.Legend(title="Crypto")),
    strokeDash=alt.StrokeDash("Type:N", legend=alt.Legend(title="Tipe")),
    tooltip=["Date:T", "Crypto:N", "Close:Q", "Type:N"]
).properties(
    width=1000,
    height=400
).interactive()

st.altair_chart(chart, use_container_width=True)

# --- METRIK TERAKHIR ---
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

# --- RANKING DENGAN LOGO ---
st.subheader("🏆 Ranking Crypto")
logo_dict = {
    "BTC-USD": "https://assets.coingecko.com/coins/images/1/large/bitcoin.png",
    "ETH-USD": "https://assets.coingecko.com/coins/images/279/large/ethereum.png",
    "BNB-USD": "https://assets.coingecko.com/coins/images/825/large/binance-coin-logo.png",
    "ADA-USD": "https://assets.coingecko.com/coins/images/975/large/cardano.png",
    "SOL-USD": "https://assets.coingecko.com/coins/images/4128/large/solana.png",
    "XRP-USD": "https://assets.coingecko.com/coins/images/44/large/xrp-symbol-white-128.png",
    "DOGE-USD": "https://assets.coingecko.com/coins/images/5/large/dogecoin.png",
    "DOT-USD": "https://assets.coingecko.com/coins/images/12171/large/polkadot.png",
    "LTC-USD": "https://assets.coingecko.com/coins/images/2/large/litecoin.png",
    "LINK-USD": "https://assets.coingecko.com/coins/images/877/large/chainlink-new-logo.png",
    "USDT-USD": "https://assets.coingecko.com/coins/images/325/large/Tether-logo.png",
    "PEPE24478-USD": "https://assets.coingecko.com/coins/images/29850/large/pepe-token.jpeg",
    "SHIB-USD": "https://assets.coingecko.com/coins/images/11939/large/shiba.png"
}

ranking = last_prices.sort_values(ascending=False)
ranking_df = pd.DataFrame({
    "Ticker": ranking.index,
    "Harga (USD)": ranking.values,
    "Nama": [
        list(crypto_dict.keys())[list(crypto_dict.values()).index(t)] if t in crypto_dict.values() else t
        for t in ranking.index
    ],
    "Logo": [logo_dict.get(t, "") for t in ranking.index]
})

for idx, row in ranking_df.iterrows():
    col1, col2, col3, col4 = st.columns([1, 1.5, 3, 3])
    with col1:
        st.markdown(f"**#{idx+1}**")
    with col2:
        if row["Logo"].startswith("http"):
            st.image(row["Logo"], width=32)

    with col3:
        st.markdown(f"**{row['Nama']} ({row['Ticker'].replace('-USD','')})**")
    with col4:
        st.markdown(f"${row['Harga (USD)']:,.2f}")

# --- FOOTER ---
st.markdown("---")
st.caption(
    """
    🚧 Dibuat dengan Streamlit · Prediksi menggunakan ARIMA · Data dari Yahoo Finance
    """
)

