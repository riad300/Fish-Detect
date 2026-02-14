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
import base64

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="FishVision Pro",
    page_icon="🐟",
    layout="wide"
)

# ================= BACKGROUND IMAGE =================
def get_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64("assets/fish_bg.png")

st.markdown(f"""
<style>
.stApp {{
    background:
    linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.85)),
    url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: white;
}}

.hero {{
    background: rgba(255,255,255,0.08);
    padding: 40px;
    border-radius: 25px;
    backdrop-filter: blur(20px);
    box-shadow: 0 0 60px rgba(0,255,255,0.25);
    text-align:center;
    margin-bottom: 30px;
}}

.glass {{
    background: rgba(255,255,255,0.07);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(18px);
    box-shadow: 0 0 40px rgba(0,255,255,0.15);
    margin-bottom: 20px;
}}

.stButton>button {{
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    border-radius: 30px;
    border:none;
    color:white;
    font-weight:bold;
    padding:0.6rem 2rem;
}}

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg,#0f2027,#203a43);
}}

footer {{visibility:hidden;}}
</style>
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

# ================= CLASS NAMES =================
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

    st.markdown("""
    <div class='hero'>
        <h1>🐟 FishVision Pro</h1>
        <p>AI Powered Fish Classification SaaS Platform</p>
    </div>
    """, unsafe_allow_html=True)

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

    st.markdown("<div class='hero'><h2>AI Fish Detection Engine</h2></div>", unsafe_allow_html=True)

    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, use_container_width=True)

        img = img.resize((224,224))
        arr = np.expand_dims(np.array(img),0)
        arr = tf.keras.applications.efficientnet_v2.preprocess_input(arr)

        pred = model.predict(arr)[0]
        top3 = pred.argsort()[-3:][::-1]

        st.subheader("🔎 Top 3 Predictions")

        for i in top3:
            fish = class_names[i]
            confidence = float(pred[i]*100)

            st.markdown(f"""
            <div class='glass'>
                <h2>{fish}</h2>
                <p>Confidence: {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(int(confidence))

        best_fish = class_names[top3[0]]
        best_conf = float(pred[top3[0]]*100)

        if best_conf > 60:
            save_history(st.session_state.user,best_fish,best_conf)

        report = f"""
FishVision AI Report
User: {st.session_state.user}
Prediction: {best_fish}
Confidence: {best_conf:.2f}%
Time: {datetime.now()}
"""
        st.download_button("⬇ Download Report",report,"prediction.txt")

# ================= HISTORY =================
if page=="📊 History":

    st.markdown("<div class='hero'><h2>Personal Dashboard</h2></div>", unsafe_allow_html=True)

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

        st.bar_chart(df["Fish"].value_counts())
        st.dataframe(df,use_container_width=True)

        csv=df.to_csv(index=False).encode()
        st.download_button("⬇ Download CSV",csv,"history.csv","text/csv")

    else:
        st.info("No predictions yet.")

# ================= ADMIN =================
if page=="📈 Analytics":

    if st.session_state.role!="admin":
        st.error("Admin Only")
    else:
        st.title("Platform Analytics")
        c.execute("SELECT fish FROM history")
        data=c.fetchall()
        if data:
            fish=[d[0] for d in data]
            chart_data={f:fish.count(f) for f in set(fish)}
            st.bar_chart(chart_data)
        else:
            st.info("No Data")

# ================= MODEL INFO =================
if page=="ℹ Model Info":

    st.title("Model Overview")
    st.write("Architecture: EfficientNetV2")
    st.write("Total Classes: 21")
    st.write("Image Size: 224x224")
    st.write("Confidence Threshold: 60%")
    st.write("Deployment: Streamlit Cloud / HuggingFace")

st.markdown("<hr><center style='color:white;'>© 2026 FishVision AI | Premium SaaS Edition</center>", unsafe_allow_html=True)
