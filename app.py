import streamlit as st
import tensorflow as tf
import numpy as np
import sqlite3
import os
import hashlib
from PIL import Image
from datetime import datetime
import requests

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="🐟 Fish Classifier Pro", layout="wide")

MODEL_URL = "https://huggingface.co/spaces/riad2021/fish-classifier/resolve/main/fish_classifier_final.keras"
MODEL_PATH = "fish_classifier_final.keras"
DB_PATH = "database.db"

class_names = [
    "Baim","Bata","Batasio(tenra)","Chitul","Croaker(Poya)",
    "Hilsha","Kajoli","Meni","Pabda","Poli",
    "Puti","Rita","Rui","Rupchada","Silver Carp",
    "Telapiya","carp","kaikka","koi","koral","shrimp"
]

# ===============================
# DATABASE SETUP (SQLite)
# ===============================
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT,
    role TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    fish TEXT,
    confidence REAL,
    time TEXT
)
""")
conn.commit()

# ===============================
# SECURITY
# ===============================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username, password, role="user"):
    try:
        c.execute("INSERT INTO users VALUES (?, ?, ?)",
                  (username, hash_password(password), role))
        conn.commit()
        return True
    except:
        return False

def verify_user(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, hash_password(password)))
    return c.fetchone()

def save_history(username, fish, confidence):
    c.execute("INSERT INTO history (username, fish, confidence, time) VALUES (?, ?, ?, ?)",
              (username, fish, confidence,
               datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()

# Create default admin if not exists
c.execute("SELECT * FROM users WHERE username='admin'")
if not c.fetchone():
    add_user("admin", "admin123", "admin")

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ===============================
# SESSION
# ===============================
if "user" not in st.session_state:
    st.session_state.user = None
if "role" not in st.session_state:
    st.session_state.role = None

# ===============================
# LOGIN / SIGNUP
# ===============================
if not st.session_state.user:

    st.title("🔐 Fish Classifier Pro Login")

    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            user = verify_user(u, p)
            if user:
                st.session_state.user = u
                st.session_state.role = user[2]
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        new_u = st.text_input("New Username")
        new_p = st.text_input("New Password", type="password")
        if st.button("Create Account"):
            if add_user(new_u, new_p):
                st.success("Account created! Login now.")
            else:
                st.warning("Username already exists.")

    st.stop()

# ===============================
# SIDEBAR
# ===============================
st.sidebar.write(f"👤 Logged in as: {st.session_state.user}")
if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

page = st.sidebar.selectbox("Navigate", 
                            ["Predict", "History", "Analytics", "Model Info"])

# ===============================
# PREDICT PAGE
# ===============================
if page == "Predict":

    st.title("🐟 Fish Species Prediction")

    uploaded = st.file_uploader("Upload Fish Image", type=["jpg","jpeg","png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_column_width=True)

        img = img.resize((224,224))
        arr = np.expand_dims(np.array(img), 0)
        arr = tf.keras.applications.efficientnet_v2.preprocess_input(arr)

        pred = model.predict(arr)
        idx = np.argmax(pred)
        confidence = float(pred[0][idx]*100)

        if confidence < 60:
            st.warning("⚠ Low confidence prediction. Try clearer image.")
        else:
            fish = class_names[idx]
            st.success(f"Prediction: {fish}")
            st.info(f"Confidence: {confidence:.2f}%")
            save_history(st.session_state.user, fish, confidence)

# ===============================
# HISTORY PAGE
# ===============================
if page == "History":
    st.title("📊 Your Prediction History")

    c.execute("SELECT fish, confidence, time FROM history WHERE username=?",
              (st.session_state.user,))
    rows = c.fetchall()

    if rows:
        st.table(rows)
    else:
        st.info("No predictions yet.")

# ===============================
# ADMIN ANALYTICS
# ===============================
if page == "Analytics":

    if st.session_state.role != "admin":
        st.error("Admin access only.")
    else:
        st.title("📈 Admin Dashboard")

        c.execute("SELECT fish FROM history")
        data = c.fetchall()

        if data:
            fish_list = [d[0] for d in data]
            st.bar_chart({f: fish_list.count(f) for f in set(fish_list)})
        else:
            st.info("No data available.")

# ===============================
# MODEL INFO
# ===============================
if page == "Model Info":
    st.title("📌 Model Information")
    st.write("Architecture: EfficientNetV2")
    st.write("Classes: 21")
    st.write("Image Size: 224x224")
    st.write("Confidence Threshold: 60%")
    st.write("Deployment: Streamlit Cloud / HuggingFace")
