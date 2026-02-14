import streamlit as st
import tensorflow as tf
import numpy as np
import sqlite3
import hashlib
import os
from PIL import Image
from datetime import datetime
import requests
import pandas as pd

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="FishVision Pro",
    page_icon="🐟",
    layout="wide"
)

# ================= PREMIUM UI =================
st.markdown("""
<style>

/* Animated Gradient */
.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1c1c1c);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: white;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Glass Card */
.glass {
    background: rgba(255,255,255,0.08);
    padding: 30px;
    border-radius: 20px;
    backdrop-filter: blur(20px);
    box-shadow: 0 0 40px rgba(0,255,255,0.15);
    transition: 0.3s;
}
.glass:hover {
    transform: translateY(-5px);
    box-shadow: 0 0 60px rgba(0,255,255,0.35);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    border-radius: 30px;
    border: none;
    color: white;
    font-weight: bold;
    padding: 0.6rem 1.5rem;
}
.stButton>button:hover {
    box-shadow: 0 0 20px #00c6ff;
    transform: scale(1.05);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141E30, #243B55);
}

/* Metric Glow */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0 0 20px rgba(0,255,255,0.15);
}

/* Watermark */
.watermark {
    position: fixed;
    bottom: 10px;
    right: 20px;
    opacity: 0.08;
    font-size: 60px;
    font-weight: 800;
}

footer {visibility:hidden;}

</style>

<div class="watermark">FishVision AI</div>
""", unsafe_allow_html=True)

# ================= MODEL =================
MODEL_URL = "https://huggingface.co/spaces/riad2021/fish-classifier/resolve/main/fish_classifier_final.keras"
MODEL_PATH = "fish_classifier_final.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ===== ORIGINAL 21 CLASS =====
class_names = [
    "Baim","Bata","Batasio (Tenra)","Chitul","Croaker (Poya)",
    "Hilsha","Kajoli","Meni","Pabda","Poli",
    "Puti","Rita","Rui","Rupchada","Silver Carp",
    "Telapiya","Carp","Kaikka","Koi","Koral","Shrimp"
]

# ================= DATABASE =================
conn = sqlite3.connect("database.db", check_same_thread=False)
c = conn.cursor()

c.execute("""CREATE TABLE IF NOT EXISTS users
             (username TEXT PRIMARY KEY, password TEXT, role TEXT)""")

c.execute("""CREATE TABLE IF NOT EXISTS history
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              username TEXT,
              fish TEXT,
              confidence REAL,
              time TEXT)""")
conn.commit()

def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()

def add_user(u,p,role="user"):
    try:
        c.execute("INSERT INTO users VALUES (?,?,?)",
                  (u,hash_password(p),role))
        conn.commit()
        return True
    except:
        return False

def verify_user(u,p):
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (u,hash_password(p)))
    return c.fetchone()

def save_history(u,f,cfg):
    c.execute("INSERT INTO history (username,fish,confidence,time) VALUES (?,?,?,?)",
              (u,f,cfg,datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()

# Default Admin
c.execute("SELECT * FROM users WHERE username='admin'")
if not c.fetchone():
    add_user("admin","admin123","admin")

# ================= SESSION =================
if "user" not in st.session_state:
    st.session_state.user=None
if "role" not in st.session_state:
    st.session_state.role=None

# ================= LOGIN =================
if not st.session_state.user:

    st.markdown("<div class='glass'><h1 style='text-align:center;'>🐟 FishVision Pro</h1><p style='text-align:center;'>AI Powered Fish Classification SaaS</p></div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔐 Login","📝 Signup"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            user = verify_user(u,p)
            if user:
                st.session_state.user=u
                st.session_state.role=user[2]
                st.success("Welcome Back 🚀")
                st.rerun()
            else:
                st.error("Invalid Credentials")

    with tab2:
        nu = st.text_input("New Username")
        npw = st.text_input("New Password", type="password")
        if st.button("Create Account"):
            if add_user(nu,npw):
                st.success("Account Created!")
            else:
                st.warning("Username Exists")

    st.stop()

# ================= SIDEBAR =================
st.sidebar.markdown("## 🐟 FishVision AI")
st.sidebar.write(f"👤 {st.session_state.user}")

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

page = st.sidebar.radio("Navigation",
                        ["🏠 Predict","📊 History","📈 Analytics","ℹ Model Info"])

# ================= PREDICT =================
if page=="🏠 Predict":

    st.title("AI Fish Species Detection")

    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, use_container_width=True)

        img = img.resize((224,224))
        arr = np.expand_dims(np.array(img),0)
        arr = tf.keras.applications.efficientnet_v2.preprocess_input(arr)

        pred = model.predict(arr)
        idx = np.argmax(pred)
        confidence = float(pred[0][idx]*100)
        fish = class_names[idx]

        st.progress(int(confidence))

        if confidence < 60:
            st.warning("Low confidence. Try clearer image.")
        else:
            st.markdown(f"""
            <div class='glass'>
                <h1 style='text-align:center;font-size:40px;'>{fish}</h1>
                <p style='text-align:center;font-size:22px;'>Confidence: {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            save_history(st.session_state.user,fish,confidence)

# ================= HISTORY =================
if page=="📊 History":

    st.title("📊 Personal Dashboard")

    c.execute("SELECT fish,confidence,time FROM history WHERE username=?",
              (st.session_state.user,))
    rows=c.fetchall()

    if rows:
        df=pd.DataFrame(rows,columns=["Fish","Confidence","Time"])

        col1,col2,col3=st.columns(3)
        col1.metric("Total Predictions",len(df))
        col2.metric("Average Confidence",
                    f"{round(df['Confidence'].mean(),2)}%")
        col3.metric("Unique Fish",df["Fish"].nunique())

        st.divider()
        st.dataframe(df,use_container_width=True)

        csv=df.to_csv(index=False).encode()
        st.download_button("⬇ Download CSV",csv,"history.csv","text/csv")

    else:
        st.info("No predictions yet.")

# ================= ADMIN ANALYTICS =================
if page=="📈 Analytics":

    if st.session_state.role!="admin":
        st.error("Admin Only Access")
    else:
        st.title("📈 Platform Analytics")

        c.execute("SELECT fish FROM history")
        data=c.fetchall()

        if data:
            fish=[d[0] for d in data]
            chart_data={f:fish.count(f) for f in set(fish)}
            st.bar_chart(chart_data)
        else:
            st.info("No Data Available")

# ================= MODEL INFO =================
if page=="ℹ Model Info":

    st.title("Model Overview")
    st.write("Architecture: EfficientNetV2")
    st.write("Total Classes: 21")
    st.write("Image Size: 224x224")
    st.write("Confidence Threshold: 60%")
    st.write("Deployment: Streamlit Cloud / HuggingFace")

st.markdown("<hr><center style='color:white;'>© 2026 FishVision AI | Premium SaaS Edition</center>", unsafe_allow_html=True)
