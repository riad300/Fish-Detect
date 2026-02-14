import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from PIL import Image
import hashlib
from datetime import datetime

st.set_page_config(page_title="Fish Species Classifier Pro", page_icon="🐟")

MODEL_PATH = "fish_classifier_final.keras"

# ---------- Utility ----------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists("users.csv"):
        pd.DataFrame(columns=["username", "password", "role"]).to_csv("users.csv", index=False)
    return pd.read_csv("users.csv")

def save_user(username, password, role="user"):
    df = load_users()
    new_user = pd.DataFrame([[username, hash_password(password), role]],
                            columns=["username","password","role"])
    df = pd.concat([df, new_user])
    df.to_csv("users.csv", index=False)

def load_history():
    if not os.path.exists("history.csv"):
        pd.DataFrame(columns=["username","fish","confidence","time"]).to_csv("history.csv", index=False)
    return pd.read_csv("history.csv")

def save_history(username, fish, confidence):
    df = load_history()
    new_row = pd.DataFrame([[username, fish, confidence, datetime.now()]],
                           columns=["username","fish","confidence","time"])
    df = pd.concat([df, new_row])
    df.to_csv("history.csv", index=False)

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

class_names = [
    "Baim","Bata","Batasio(tenra)","Chitul","Croaker(Poya)",
    "Hilsha","Kajoli","Meni","Pabda","Poli",
    "Puti","Rita","Rui","Rupchada","Silver Carp",
    "Telapiya","carp","kaikka","koi","koral","shrimp"
]

# ---------- Session ----------
if "user" not in st.session_state:
    st.session_state.user = None
if "role" not in st.session_state:
    st.session_state.role = None

# ---------- Authentication ----------
def login():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        users = load_users()
        hashed = hash_password(password)
        user = users[(users.username==username) & (users.password==hashed)]
        if not user.empty:
            st.session_state.user = username
            st.session_state.role = user.iloc[0]["role"]
            st.success("Login successful!")
        else:
            st.error("Invalid credentials")

def signup():
    st.subheader("Signup")
    username = st.text_input("New Username")
    password = st.text_input("New Password", type="password")
    if st.button("Create Account"):
        save_user(username, password)
        st.success("Account created!")

if not st.session_state.user:
    menu = st.sidebar.selectbox("Menu", ["Login","Signup"])
    if menu=="Login":
        login()
    else:
        signup()
    st.stop()

# ---------- Main App ----------
st.sidebar.write(f"👤 Logged in as: {st.session_state.user}")
if st.sidebar.button("Logout"):
    st.session_state.user=None
    st.experimental_rerun()

page = st.sidebar.selectbox("Navigate", ["Predict","History","Analytics","Model Info"])

# ---------- Predict ----------
if page=="Predict":
    st.title("🐟 Fish Species Classifier Pro")
    uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_column_width=True)
        img = img.resize((224,224))
        img_array = np.expand_dims(np.array(img),0)
        img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

        pred = model.predict(img_array)
        idx = np.argmax(pred)
        confidence = float(pred[0][idx]*100)

        if confidence < 60:
            st.warning("Low confidence prediction. Try clearer image.")
        else:
            fish = class_names[idx]
            st.success(f"Prediction: {fish}")
            st.info(f"Confidence: {confidence:.2f}%")
            save_history(st.session_state.user, fish, confidence)

# ---------- History ----------
if page=="History":
    st.title("Prediction History")
    df = load_history()
    user_df = df[df.username==st.session_state.user]
    st.dataframe(user_df)

# ---------- Analytics (Admin Only) ----------
if page=="Analytics":
    if st.session_state.role!="admin":
        st.warning("Admin access only")
    else:
        st.title("Admin Analytics Dashboard")
        df = load_history()
        st.dataframe(df)
        st.bar_chart(df["fish"].value_counts())

# ---------- Model Info ----------
if page=="Model Info":
    st.title("Model Information")
    st.write("Architecture: EfficientNetV2")
    st.write("Dataset Size: 583 Images")
    st.write("Classes: 21")
    st.write("Image Size: 224x224")
