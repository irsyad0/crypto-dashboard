import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
from PIL import Image
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA
from requests import get # Untuk mengambil data live dari CoinGecko

# --- PAGE CONFIG ---
st.set_page_config(page_title="Crypto Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS (DARK THEME) ---
st.markdown("""
<style>
/* Global styles for tables - Dark Theme */
.dataframe, .prediction-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9em;
    text-align: left;
    margin-bottom: 20px;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    background: #1a1a2e;
}

.dataframe th, .prediction-table th {
    background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
    color: #e94560;
    padding: 15px;
    border: none;
    text-align: left;
    font-weight: 600;
    font-size: 0.95em;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
}

/* Specific styles for prediction table header */
.prediction-table th {
    background: linear-gradient(135deg, #2d1b69 0%, #11998e 100%);
    color: #f38ba8;
}

.dataframe td, .prediction-table td {
    padding: 12px 15px;
    border: none;
    border-bottom: 1px solid #2a2a3e;
    transition: all 0.3s ease;
    color: #cdd6f4;
}

.dataframe tr:nth-child(even), .prediction-table tr:nth-child(even) {
    background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
}

.dataframe tr:nth-child(odd), .prediction-table tr:nth-child(odd) {
    background: linear-gradient(135deg, #181825 0%, #1e1e2e 100%);
}

.dataframe tr:hover, .prediction-table tr:hover {
    background: linear-gradient(135deg, #313244 0%, #45475a 100%);
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(233, 69, 96, 0.2);
}

/* Common styles for changes and logos */
.positive-change {
    color: #a6e3a1;
    font-weight: bold;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}
.negative-change {
    color: #f38ba8;
    font-weight: bold;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}
.crypto-logo {
    vertical-align: middle;
    margin-right: 8px;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    box-shadow: 0 3px 6px rgba(0,0,0,0.4);
}
.arrow-up {
    color: #a6e3a1;
}
.arrow-down {
    color: #f38ba8;
}

/* Custom metric card styling - Dark Theme */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 20px;
    border-radius: 15px;
    color: #cdd6f4;
    text-align: center;
    box-shadow: 0 10px 20px rgba(0,0,0,0.4);
    margin-bottom: 10px;
    transition: transform 0.3s ease;
    border: 1px solid #313244;
}
.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 30px rgba(233, 69, 96, 0.2);
    border-color: #e94560;
}
.crypto-metric-logo {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    margin-bottom: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
}
.metric-name {
    font-size: 1.1em;
    font-weight: 600;
    margin-bottom: 8px;
    color: #f9e2af;
}
.metric-price {
    font-size: 1.6em;
    font-weight: bold;
    margin-bottom: 8px;
    color: #cdd6f4;
}
.metric-change {
    font-size: 1.1em;
    font-weight: 600;
}

/* Dark theme for general Streamlit elements */
.stApp {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
    color: #cdd6f4;
}

/* Sidebar dark styling */
.css-1d391kg {
    background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
}

/* Dark styling for headers */
h1, h2, h3, h4, h5, h6 {
    color: #f9e2af !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

/* Dark styling for regular text */
p, div, span {
    color: #cdd6f4;
}
</style>
""", unsafe_allow_html=True)


# --- HEADER (CUSTOM IMAGE REPLACES TEXT HEADER) ---
try:
    # Path gambar, pastikan ini benar di sistem Anda
    custom_image = Image.open(r"gambar.png")
    st.image(custom_image, use_container_width=True)
except Exception as e:
    st.warning(f"Gagal memuat custom image. Pastikan path gambar sudah benar: {e}")

st.markdown("---")

# --- SIDEBAR FILTERS ---
with st.sidebar:
    st.header("⚙️ Pengaturan Dashboard")
    crypto_dict = {
        "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD",
        "Cardano": "ADA-USD", "Solana": "SOL-USD", 
        "Dogecoin": "DOGE-USD", "Polkadot": "DOT-USD", "Litecoin": "LTC-USD",
        "Chainlink": "LINK-USD", "Tether": "USDT-USD", "TRON": "TRX-USD"
    }
    pilihan = st.multiselect(
        "Pilih crypto untuk grafik dan prediksi:", list(crypto_dict.keys()), default=["Bitcoin", "Ethereum"]
    )
    jumlah_hari = st.slider(
        "Prediksi berapa hari ke depan?", 1, 30, 1 # Default 7 hari untuk prediksi
    )

if not pilihan:
    st.warning("Silakan pilih minimal satu crypto untuk ditampilkan pada grafik dan prediksi.")
    st.stop()

tickers = [crypto_dict[n] for n in pilihan]

# --- CACHING FUNGSI PENGAMBIL DATA ---
@st.cache_data(ttl=3600) # Data di-cache selama 1 jam
def ambil_data_historis(tickers_list):
    """Mengambil data harga historis dari Yahoo Finance."""
    start_date = "2015-01-01"
    end_date = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    try:
        data = yf.download(tickers_list, start=start_date, end=end_date)["Close"]
        # Isi nilai NaN di awal dengan 0 (jika belum ada data historis)
        for col in data.columns:
            first_valid = data[col].first_valid_index()
            if first_valid:
                data.loc[:first_valid, col] = 0
            else:
                data[col] = 0
        return data.fillna(0)
    except Exception as e:
        st.error(f"Gagal mengambil data historis dari Yahoo Finance: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600) # Data live di-cache selama 10 menit
def ambil_data_live_coingecko():
    """Mengambil data live (harga, perubahan persentase, kapitalisasi pasar) dari CoinGecko."""
    try:
        # Mengambil data 100 crypto teratas berdasarkan market cap
        url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=false&price_change_percentage=24h"
        response = get(url)
        response.raise_for_status() # Akan memunculkan HTTPError untuk status kode 4xx/5xx
        return response.json()
    except Exception as e:
        st.error(f"Gagal mengambil data live dari CoinGecko: {e}")
        return []

# --- AMBIL DATA ---
with st.spinner("Mengambil data historis dari Yahoo Finance..."):
    data_harga = ambil_data_historis(tickers)

if data_harga.empty:
    st.warning("Data historis tidak tersedia untuk crypto yang dipilih. Mohon periksa pilihan Anda atau coba lagi nanti.")
    st.stop()

with st.spinner("Mengambil data live dari CoinGecko..."):
    live_crypto_data = ambil_data_live_coingecko()

st.markdown(
    f"📅 Data historis dari **{data_harga.index.min().strftime('%d-%m-%Y')}** "
    f"sampai **{data_harga.index.max().strftime('%d-%m-%Y')}**"
)

# --- MODEL PREDIKSI (ARIMA) ---
st.markdown("---")

prediksi_df_raw = pd.DataFrame() # Data prediksi mentah
last_prices_dict = {} # Harga terakhir untuk perhitungan persentase prediksi

for t in tickers:
    df_ts = data_harga[t].reset_index().rename(columns={t: "Close"}).set_index('Date')
    df_ts = df_ts.asfreq('D').ffill().bfill().dropna()

    if df_ts.empty or df_ts['Close'].isnull().any() or len(df_ts) < 60:
        st.warning(f"Data tidak cukup atau bermasalah untuk {t} untuk prediksi. Dilewati.")
        continue

    try:
        # Uji stasioneritas 
        model = ARIMA(df_ts['Close'], order=(1,1,2))  
        model_fit = model.fit()
        
        future_pred = model_fit.forecast(steps=jumlah_hari)
        future_pred = np.where(future_pred < 0, 0, future_pred)  

        # Buat tanggal prediksi
        future_dates = pd.date_range(start=df_ts.index[-1] + timedelta(days=1), periods=jumlah_hari, freq='D')
        prediksi_df_raw[t] = pd.Series(future_pred, index=future_dates)
        last_prices_dict[t] = df_ts['Close'].iloc[-1]

    except Exception as e:
        st.warning(f"❌ Prediksi gagal untuk {t}: {e}")

# --- VISUALISASI GRAFIK HARGA & PREDIKSI ---
st.subheader("📈 Grafik Harga Historis & Prediksi")

# Siapkan data historis untuk plotting
df_hist = data_harga.reset_index().melt(
    id_vars="Date", var_name="Crypto", value_name="Close"
)
df_hist["Type"] = "Historis"

# Siapkan data prediksi untuk plotting
df_pred_plot = pd.DataFrame()
if not prediksi_df_raw.empty:
    df_pred_plot = prediksi_df_raw.reset_index().melt(
        id_vars="index", var_name="Crypto", value_name="Close"
    ).rename(columns={"index": "Date"})
    df_pred_plot["Type"] = "Prediksi"

# Gabungkan data historis dan prediksi
df_all = pd.concat([df_hist, df_pred_plot])

# Buat chart Altair dengan dark theme
chart = alt.Chart(df_all).mark_line(size=3).encode(
    x=alt.X("Date:T", title="Waktu"),
    y=alt.Y("Close:Q", title="Harga (USD)"),
    color=alt.Color("Crypto:N", legend=alt.Legend(title="Crypto"), scale=alt.Scale(scheme='dark2')),
    strokeDash=alt.StrokeDash("Type:N", legend=alt.Legend(title="Tipe")),
    tooltip=[
        alt.Tooltip("Date:T", title="Tanggal"),
        alt.Tooltip("Crypto:N", title="Crypto"),
        alt.Tooltip("Close:Q", title="Harga", format="$.2f"),
        alt.Tooltip("Type:N", title="Tipe Data")
    ]
).properties(
    title="Pergerakan Harga Cryptocurrency dan Prediksi",
    background='#1a1a2e'
).configure_title(
    color='#f9e2af',
    fontSize=16
).configure_axis(
    labelColor='#cdd6f4',
    titleColor='#f9e2af',
    gridColor='#313244'
).configure_legend(
    labelColor='#cdd6f4',
    titleColor='#f9e2af'
).interactive() # Memungkinkan zoom dan pan

st.altair_chart(chart, use_container_width=True)

# --- TABEL PREDIKSI HARGA ---
st.markdown("---")
st.subheader("📆 Detail Prediksi Harga")
st.write(f"Prediksi harga untuk {jumlah_hari} hari ke depan dari harga terakhir.")

if prediksi_df_raw.empty:
    st.info("Tidak ada prediksi yang tersedia untuk ditampilkan.")
else:
    prediksi_formatted_data = []
    for date_idx, date in enumerate(prediksi_df_raw.index):
        row_data = {"Tanggal": date.strftime('%d-%m-%Y')}
        for ticker in prediksi_df_raw.columns:
            predicted_price = prediksi_df_raw.loc[date, ticker]
            last_price = last_prices_dict.get(ticker, 0) # Ambil harga terakhir dari dictionary
            
            # Hitung persentase perubahan dari harga terakhir
            percentage_change = 0
            if last_price > 0:
                percentage_change = ((predicted_price - last_price) / last_price) * 100
            
            # Tentukan panah
            arrow_icon = ""
            if percentage_change > 0:
                arrow_icon = "⬆️" # Panah atas
            elif percentage_change < 0:
                arrow_icon = "⬇️" # Panah bawah
            
            # Tambahkan styling CSS untuk warna panah
            color_class = "positive-change" if percentage_change >= 0 else "negative-change"
            
            # Menggunakan f-string yang lebih sederhana untuk menghindari masalah spasi/indentasi
            row_data[ticker] = (
                f"<div style='display: flex; align-items: center; justify-content: space-between;'>"
                f"<span>${predicted_price:,.2f}</span>"
                f"<span class='{color_class}' style='font-size: 0.8em; margin-left: 5px;'>"
                f"{arrow_icon} {percentage_change:.2f}%"
                f"</span></div>"
            )
        prediksi_formatted_data.append(row_data)

    # Buat HTML untuk tabel prediksi
    # Pastikan tidak ada indentasi yang tidak perlu pada awal baris HTML
    table_pred_html_content = ""
    for row_data_item in prediksi_formatted_data:
        table_pred_html_content += "<tr>"
        for key, value in row_data_item.items():
            table_pred_html_content += f"<td>{value}</td>"
        table_pred_html_content += "</tr>"

    table_pred_html = f"""
<table class='prediction-table'>
    <thead>
        <tr>
            <th>Tanggal</th>
"""
    for ticker in prediksi_df_raw.columns:
        display_name = next((k for k, v in crypto_dict.items() if v == ticker), ticker.replace('-USD',''))
        table_pred_html += f"<th>{display_name}</th>"
    table_pred_html += f"""
        </tr>
    </thead>
    <tbody>
        {table_pred_html_content}
    </tbody>
</table>
"""
    st.markdown(table_pred_html, unsafe_allow_html=True)


# --- METRIK HARGA TERKINI ---
st.markdown("---")
st.subheader("📊 Harga Terkini")

if not live_crypto_data:
    st.warning("Tidak dapat menampilkan metrik harga terkini karena gagal mengambil data live.")
else:
    # Buat dictionary untuk mapping ticker Yahoo Finance ke ID CoinGecko
    yfinance_to_coingecko_id = {v: k for k, v in crypto_dict.items()}
    
    # Filter data live hanya untuk crypto yang dipilih di sidebar
    filtered_live_data = [
        item for item in live_crypto_data 
        if any(item['id'] == yfinance_to_coingecko_id[t].lower() for t in tickers) # Konversi ke lowercase untuk ID CoinGecko
    ]

    if filtered_live_data:
        max_per_row = 5
        for i in range(0, len(filtered_live_data), max_per_row):
            cols = st.columns(max_per_row) # Buat kolom sesuai jumlah crypto per baris
            for j, crypto_item in enumerate(filtered_live_data[i:i+max_per_row]):
                with cols[j]:
                    price_change = crypto_item.get('price_change_percentage_24h', 0)
                    change_sign = "+" if price_change > 0 else ""
                    change_color = "#a6e3a1" if price_change >= 0 else "#f38ba8"
                    
                    # Custom metric card with logo
                    metric_html = f"""
                    <div class="metric-card">
                        <img src="{crypto_item['image']}" class="crypto-metric-logo" alt="{crypto_item['name']} Logo">
                        <div class="metric-name">{crypto_item['name']}</div>
                        <div class="metric-price">${crypto_item['current_price']:,.2f}</div>
                        <div class="metric-change" style="color: {change_color};">
                            {change_sign}{price_change:.2f}%
                        </div>
                    </div>
                    """
                    st.markdown(metric_html, unsafe_allow_html=True)
    else:
        st.info("Tidak ada data harga terkini untuk crypto yang Anda pilih di sidebar.")


# --- TABEL RANKING DENGAN PERSENTASE DAN LOGO ---
st.markdown("---")
st.subheader("🏆 Ranking Crypto (Perubahan 24 Jam)")

if not live_crypto_data:
    st.warning("Tidak dapat menampilkan ranking karena gagal mengambil data live.")
else:
    ranking_data = []
    for item in live_crypto_data:
        if 'price_change_percentage_24h' in item and item['price_change_percentage_24h'] is not None:
            ranking_data.append({
                "name": item['name'],
                "symbol": item['symbol'].upper(),
                "current_price": item['current_price'],
                "price_change_percentage_24h": item['price_change_percentage_24h'],
                "image": item['image'], # URL logo dari CoinGecko
                "market_cap": item['market_cap']
            })
    
    if ranking_data:
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values(by="price_change_percentage_24h", ascending=False).reset_index(drop=True)
        ranking_df["Rank"] = ranking_df.index + 1

        st.write("Daftar 20 Cryptocurrency Teratas berdasarkan Perubahan Harga 24 Jam:")
        
        display_df = ranking_df.head(20)

        # Buat HTML untuk tabel ranking
        # Memastikan tidak ada indentasi yang tidak perlu pada awal baris HTML
        table_ranking_html_content = ""
        for index, row in display_df.iterrows():
            price_change_value = row['price_change_percentage_24h']
            
            arrow_icon = ""
            if price_change_value > 0:
                arrow_icon = "<span class='arrow-up'>⬆️</span>"
            elif price_change_value < 0:
                arrow_icon = "<span class='arrow-down'>⬇️</span>"
            
            price_change_class = "positive-change" if price_change_value >= 0 else "negative-change"

            table_ranking_html_content += f"""
<tr>
    <td>{row['Rank']}</td>
    <td><img src="{row['image']}" class="crypto-logo" alt="{row['name']} Logo"></td>
    <td><strong>{row['name']}</strong> ({row['symbol']})</td>
    <td>${row['current_price']:,.2f}</td>
    <td class="{price_change_class}">{arrow_icon} {price_change_value:.2f}%</td>
    <td>${row['market_cap']:,.0f}</td>
</tr>
"""
        table_ranking_html = f"""
<table class='dataframe'>
    <thead>
        <tr>
            <th>Rank</th>
            <th></th>
            <th>Nama Crypto</th>
            <th>Harga (USD)</th>
            <th>Perubahan 24h (%)</th>
            <th>Kapitalisasi Pasar</th>
        </tr>
    </thead>
    <tbody>
        {table_ranking_html_content}
    
</table>
"""
        st.markdown(table_ranking_html, unsafe_allow_html=True)
    else:
        st.info("Tidak ada data ranking yang tersedia.")


# --- FOOTER ---
st.markdown("---")
st.caption(
    """
    🚧 Dibuat dengan Streamlit · Data Historis dari Yahoo Finance · Data Live & Logo dari CoinGecko API · Prediksi menggunakan ARIMA
    """
)