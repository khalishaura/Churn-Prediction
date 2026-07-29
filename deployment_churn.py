# ============================================================
# BLOK 1: Import & konfigurasi
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediksi Churn Pelanggan",
                    page_icon="📉", layout="centered")

THRESHOLD = 0.5

# Label untuk fitur numerik
LABEL = {
    "Tenure": "Lama berlangganan",
    "WarehouseToHome": "Jarak gudang ke rumah",
    "HourSpendOnApp": "Jam pakai aplikasi",
    "NumberOfDeviceRegistered": "Jumlah perangkat terdaftar",
    "NumberOfAddress": "Jumlah alamat tersimpan",
    "OrderAmountHikeFromlastYear": "Kenaikan nilai belanja",
    "CouponUsed": "Kupon dipakai",
    "OrderCount": "Jumlah pesanan",
    "DaySinceLastOrder": "Hari sejak pesanan terakhir",
    "CashbackAmount": "Nilai cashback",
}

def label_of(col):
    return LABEL.get(col, col)

def induk(col):
    """Petakan kolom hasil one-hot ke nama fitur aslinya."""
    for prefix, nama in [
        ("PreferredLoginDevice_", "Perangkat login"),
        ("PreferredPaymentMode_", "Metode pembayaran"),
        ("PreferedOrderCat_", "Kategori belanja"),
        ("SatisfactionScore_", "Skor kepuasan"),
        ("MaritalStatus_", "Status pernikahan"),
        ("CityTier_", "Tier kota"),
        ("Gender_", "Jenis kelamin"),
        ("Complain_", "Pernah komplain"),
    ]:
        if col.startswith(prefix):
            return nama
    return label_of(col)

# ============================================================
# Load model, scaler, & explainer
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("MLP.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    batas_atas_outlier = joblib.load("batas_atas_outlier.pkl")
    bg = joblib.load("shap_background.pkl")   # background SHAP (sudah di-scaling, 100 baris)

    def f_churn(X):
        return model.predict_proba(np.asarray(X))[:, 1]

    explainer = shap.Explainer(f_churn, bg)

    return model, scaler, columns, batas_atas_outlier, explainer

try:
    model, scaler, columns, batas_atas_outlier, explainer = load_artifacts()
except FileNotFoundError as e:
    st.error(f"File tidak ditemukan: {e.filename}")
    st.stop()

# ============================================================
# Header & sidebar
# ============================================================
st.title("📉 Prediksi Churn Pelanggan E-Commerce")
st.write("Isi data pelanggan di bawah ini.")

st.sidebar.header("ℹ️ Tentang aplikasi")
st.sidebar.write(
    "Aplikasi ini digunakan untuk memprediksi kemungkinan pelanggan berhenti menggunakan layanan e-commerce (churn)."
)
st.sidebar.divider()
st.sidebar.subheader("Cara pakai")
st.sidebar.markdown(
    "1. Isi data pelanggan\n"
    "2. Klik **Prediksi**\n"
    "3. Lihat hasil & faktor yang meningkatkan hasil prediksinya"
)
st.sidebar.divider()
with st.sidebar.expander("Detail teknis"):
    st.markdown(
        "- Model: Multilayer Perceptron (MLP)\n"
        "- Data latih: 5.630 pelanggan e-commerce\n"
        "- Penjelasan faktor: SHAP (Permutation Explainer)\n"
    )
st.sidebar.caption("Portfolio project · Khalisha Haura Zahra · 2026")

# ============================================================
# Form input
# ============================================================
st.subheader("Data pelanggan")
c1, c2 = st.columns(2)
with c1:
    tenure       = st.number_input("Lama berlangganan (bulan)", 0, 100, 10)
    city_tier    = st.selectbox("Tier kota", [1, 2, 3])
    warehouse    = st.number_input("Jarak gudang ke rumah (km)", 0, 50, 15)
    hour_app     = st.number_input("Jam pakai aplikasi per hari", 0, 24, 3)
    n_device     = st.number_input("Jumlah perangkat terdaftar", 1, 10, 3)
    satisfaction = st.selectbox("Skor kepuasan", [1, 2, 3, 4, 5], index=2)
    n_address    = st.number_input("Jumlah alamat tersimpan", 1, 25, 3)
with c2:
    complain     = st.selectbox("Pernah komplain?", ["Tidak", "Ya"])
    order_hike   = st.number_input("Kenaikan belanja (%)", 0, 100, 15)
    coupon       = st.number_input("Jumlah kupon dipakai", 0, 50, 1)
    order_count  = st.number_input("Jumlah pesanan", 0, 100, 2)
    day_last     = st.number_input("Hari sejak pesanan terakhir", 0, 100, 5)
    cashback     = st.number_input("Nilai cashback", 0, 500, 150)

login_device = st.selectbox("Perangkat login", ["Mobile Phone", "Computer"])
payment      = st.selectbox("Metode pembayaran",
                            ["Debit Card", "Credit Card", "E wallet", "UPI", "Cash on Delivery"])
gender       = st.selectbox("Jenis kelamin", ["Female", "Male"])
order_cat    = st.selectbox("Kategori belanja",
                            ["Fashion", "Grocery", "Laptop & Accessory",
                             "Mobile Phone", "Others"])
marital      = st.selectbox("Status pernikahan", ["Single", "Married", "Divorced"])

# ============================================================
# Preprocessing 
# ============================================================
raw = {
    "Tenure": tenure,
    "WarehouseToHome": min(warehouse, batas_atas_outlier),   
    "HourSpendOnApp": hour_app,
    "NumberOfDeviceRegistered": n_device,
    "NumberOfAddress": n_address,
    "OrderAmountHikeFromlastYear": order_hike,
    "CouponUsed": coupon,
    "OrderCount": order_count,
    "DaySinceLastOrder": day_last,
    "CashbackAmount": cashback,
}

# One-hot encoding manual
raw["PreferredLoginDevice_Mobile Phone"] = int(login_device == "Mobile Phone")
raw["CityTier_2"]   = int(city_tier == 2)
raw["CityTier_3"]   = int(city_tier == 3)
raw["Gender_Male"]  = int(gender == "Male")
raw["Complain_1"]   = int(complain == "Ya")
for k in ["Credit Card", "Debit Card", "E wallet", "UPI"]:
    raw[f"PreferredPaymentMode_{k}"] = int(payment == k)
for k in ["Grocery", "Laptop & Accessory", "Mobile Phone", "Others"]:
    raw[f"PreferedOrderCat_{k}"] = int(order_cat == k)
for k in [2, 3, 4, 5]:
    raw[f"SatisfactionScore_{k}"] = int(satisfaction == k)
for k in ["Married", "Single"]:
    raw[f"MaritalStatus_{k}"] = int(marital == k)

df = pd.DataFrame([raw])

# Jika kolom tidak cocok dengan waktu training, langsung berhenti
kurang = set(columns) - set(df.columns)
lebih  = set(df.columns) - set(columns)
if kurang or lebih:
    st.error(f"Kolom tidak cocok dengan model.\n\nKurang: {kurang}\n\nBerlebih: {lebih}")
    st.stop()

df = df[columns]  

df_scaled = pd.DataFrame(scaler.transform(df), columns=columns)

# ============================================================
# Prediksi
# ============================================================
if st.button("Prediksi", type="primary"):
    proba = model.predict_proba(df_scaled)[0, 1]
    churn = proba >= THRESHOLD

    st.divider()
    m1, m2 = st.columns(2)
    tampil_proba = f"{proba:.2%}" if proba < 0.01 else f"{proba:.1%}"
    m1.metric("Kemungkinan churn", tampil_proba)
    m2.metric("Keputusan", "BERISIKO CHURN ⚠️" if churn else "AMAN ✅")
    st.progress(float(proba))

    if churn:
        st.error("Pelanggan berisiko berhenti ➜ sebaiknya diprioritaskan untuk retensi.")
    else:
        st.success("Pelanggan diperkirakan akan bertahan.")

# ============================================================
# SHAP
# ============================================================
    def deskripsi_fitur(col):
        """Ubah nama kolom jadi kalimat pakai nilai input asli."""
        if col == "Tenure":
            return f"Lama berlangganan {tenure} bulan"
        if col == "WarehouseToHome":
            return f"Jarak gudang ke rumah {warehouse} km"
        if col == "HourSpendOnApp":
            return f"Memakai aplikasi {hour_app} jam"
        if col == "NumberOfDeviceRegistered":
            return f"{n_device} perangkat terdaftar"
        if col == "NumberOfAddress":
            return f"{n_address} alamat tersimpan"
        if col == "OrderAmountHikeFromlastYear":
            return f"Kenaikan belanja {order_hike}%"
        if col == "CouponUsed":
            return f"Memakai {coupon} kupon"
        if col == "OrderCount":
            return f"{order_count} kali pesanan"
        if col == "DaySinceLastOrder":
            return f"{day_last} hari sejak pesanan terakhir"
        if col == "CashbackAmount":
            return f"Cashback senilai {cashback}"
        if col.startswith("PreferredLoginDevice_"):
            return f"Login melalui {login_device}"
        if col.startswith("CityTier_"):
            return f"Tinggal di kota Tier {city_tier}"
        if col.startswith("Gender_"):
            return "Berjenis kelamin Pria" if gender == "Male" else "Berjenis kelamin Wanita"
        if col.startswith("Complain_"):
            return "Tidak pernah komplain" if complain == "Tidak" else "Pernah komplain"
        if col.startswith("PreferredPaymentMode_"):
            return f"Membayar dengan {payment}"
        if col.startswith("PreferedOrderCat_"):
            return f"Kategori belanja {order_cat}"
        if col.startswith("SatisfactionScore_"):
            return f"Skor kepuasan {satisfaction}"
        if col.startswith("MaritalStatus_"):
            return f"Berstatus {marital}"
        return col   

    with st.spinner("Menghitung penjelasan SHAP..."):
        sv_one = explainer(df_scaled)          
        sv_one.data = df[columns].values       
        vals = sv_one.values[0]

    st.divider()

    if churn:
        idx_urut = np.argsort(-vals)[:3]
        judul = "3 Faktor Meningkatkan Prediksi Pelanggan Churn"
    else:
        idx_urut = np.argsort(vals)[:3]
        judul = "3 Faktor Meningkatkan Prediksi Pelanggan Bertahan (Non-Churn)"

    st.subheader(judul)

    sudah_tampil = set()
    ditampilkan = 0
    for i in idx_urut:
        nama_kolom = columns[i]
        kalimat = deskripsi_fitur(nama_kolom)
        if kalimat in sudah_tampil:
            continue
        sudah_tampil.add(kalimat)

        skor = vals[i]
        arah = "meningkatkan" if skor > 0 else "menurunkan"
        ikon = "🔴" if skor > 0 else "🟢"
        st.markdown(f"{ikon} **{kalimat}** — {arah} prediksi churn dengan nilai SHAP ({skor:+.3f})")

        ditampilkan += 1
        if ditampilkan >= 3:
            break