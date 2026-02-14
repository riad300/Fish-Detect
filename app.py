import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import hashlib
from PIL import Image
from datetime import datetime

st.set_page_config(page_title="🐟Fish Species Classifier Pro", page_icon="🐟", layout="wide")

# ===== Model Loading =====
MODEL_URL = "https://huggingface.co/spaces/riad2021/fish-classifier/resolve/main/fish_classifier_final.keras"
MODEL_PATH = "fish_classifier_final.keras"

@st.cache_resource
def load_model_from_url():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            import requests
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model_from_url()

class_names = [
    "Baim","Bata","Batasio(tenra)","Chitul","Croaker(Poya)",
    "Hilsha","Kajoli","Meni","Pabda","Poli",
    "Puti","Rita","Rui","Rupchada","Silver Carp",
    "Telapiya","carp","kaikka","koi","koral","shrimp"
]

# ===== Auth + DB =====

if not os.path.exists("users.csv"):
    pd.DataFrame(columns=["username","password","role"]).to_csv("users.csv", index=False)

if not os.path.exists("history.csv"):
    pd.DataFrame(columns=["username","fish","confidence","time"]).to_csv("history.csv", index=False)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username, password, role="user"):
    df = pd.read_csv("users.csv")
    df.loc[len(df)] = [username, hash_password(password), role]
    df.to_csv("users.csv", index=False)

def verify_user(username, password):
    df = pd.read_csv("users.csv")
    h = hash_password(password)
    user = df[(df.username == username) & (df.password == h)]
    return False if user.empty else True

# Manage session
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None

# ===== Authentication =====

if not st.session_state.logged_in:
    st.title("🔐 Login or Signup")
    tab = st.tabs(["Login","Signup"])

    with tab[0]:
        user = st.text_input("Username", key="login_user")
        pwd = st.text_input("Password", type="password", key="login_pwd")
        if st.button("Login"):
            if verify_user(user, pwd):
                st.session_state.logged_in = True
                st.session_state.user = user
                df = pd.read_csv("users.csv")
                st.session_state.role = df[df.username==user].role.values[0]
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab[1]:
        new_user = st.text_input("New Username", key="signup_user")
        new_pwd = st.text_input("New Password", type="password", key="signup_pwd")
        if st.button("Signup"):
            add_user(new_user, new_pwd)
            st.success("Account created! Now log in")

    st.stop()

# ===== Main App UI =====

st.sidebar.write(f"👤 Logged in as: **{st.session_state.user}**")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.role = None
    st.experimental_rerun()

page = st.sidebar.selectbox("Navigate", ["Predict","History","Analytics","Download Report","Model Info"])

# Save history
def save_history(username, fish, confidence):
    df = pd.read_csv("history.csv")
    df.loc[len(df)] = [username, fish, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    df.to_csv("history.csv", index=False)

# ===== PREDICT =====

if page == "Predict":
    st.title("🐟 Upload Fish Image")
    uploaded = st.file_uploader("", type=["jpg","jpeg","png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img = img.resize((224,224))
        arr = np.expand_dims(np.array(img), 0)
        arr = tf.keras.applications.efficientnet_v2.preprocess_input(arr)

        pred = model.predict(arr)
        idx = np.argmax(pred, axis=1)[0]
        confidence = float(pred[0][idx]*100)

        if confidence < 60:
            st.warning("⚠ Low confidence — please upload a clearer image")
        else:
            fish = class_names[idx]
            st.success(f"Prediction: **{fish}**")
            st.write(f"Confidence: **{confidence:.2f}%**")
            save_history(st.session_state.user, fish, confidence)

# ===== HISTORY =====

if page == "History":
    st.title("📊 Your Prediction History")
    h = pd.read_csv("history.csv")
    user_hist = h[h.username == st.session_state.user]
    st.dataframe(user_hist)

# ===== ANALYTICS (Admin Only) =====

if page == "Analytics":
    if st.session_state.role != "admin":
        st.error("Admin access only")
    else:
        st.title("📈 Admin Dashboard")
        h = pd.read_csv("history.csv")
        st.write("Overall Prediction Records")
        st.dataframe(h)
        st.divider()
        st.write("Most Predicted Fish Count")
        st.bar_chart(h["fish"].value_counts())

# ===== DOWNLOAD REPORT =====

if page == "Download Report":
    st.title("📄 Download Your Report")
    h = pd.read_csv("history.csv")
    user_hist = h[h.username == st.session_state.user]
    st.dataframe(user_hist)

    if st.button("Download PDF Report"):
        import pandas as pd
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Fish Prediction Report", ln=True)

        for i,row in user_hist.iterrows():
            pdf.cell(200, 7, txt=f"{row.username} | {row.fish} | {row.confidence:.2f}% | {row.time}", ln=True)

        pdf.output("report.pdf")
        st.success("Report generated!")
        st.markdown("[Download PDF](report.pdf)")

# ===== MODEL INFO =====

if page == "Model Info":
    st.title("📌 Model Information")
    st.write("- Model: EfficientNetV2 trained on 583 images")
    st.write("- Classes: 21 Fish Species")
    st.write("- Image Size: 224×224")
    st.write("- Confidence Threshold: 60%")
