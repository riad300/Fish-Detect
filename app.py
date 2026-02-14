import streamlit as st
import sqlite3
import hashlib
import pandas as pd
import datetime
import random
import base64

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="FishVision AI",
    page_icon="🐟",
    layout="wide"
)

# -------------------------
# CUSTOM CSS (SaaS UI)
# -------------------------
st.markdown("""
<style>

/* Background Gradient */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Glass Card Effect */
.glass {
    background: rgba(255, 255, 255, 0.08);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.37);
}

/* Header Styling */
h1, h2, h3 {
    font-weight: 700;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141E30, #243B55);
}

/* Watermark */
.watermark {
    position: fixed;
    bottom: 10px;
    right: 20px;
    opacity: 0.15;
    font-size: 60px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='watermark'>FishVision AI</div>", unsafe_allow_html=True)

# -------------------------
# DATABASE
# -------------------------
conn = sqlite3.connect("fish_app.db", check_same_thread=False)
c = conn.cursor()

c.execute("""CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY,
            password TEXT)""")

c.execute("""CREATE TABLE IF NOT EXISTS history(
            username TEXT,
            fish TEXT,
            confidence REAL,
            time TEXT)""")

conn.commit()

# -------------------------
# HASH PASSWORD
# -------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# -------------------------
# SESSION
# -------------------------
if "user" not in st.session_state:
    st.session_state.user = None

# -------------------------
# AUTH SECTION
# -------------------------
def login():
    st.markdown("<h1>🔐 Login</h1>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        hashed = hash_password(password)
        c.execute("SELECT * FROM users WHERE username=? AND password=?",
                  (username, hashed))
        if c.fetchone():
            st.session_state.user = username
            st.success("Login Successful 🚀")
            st.rerun()
        else:
            st.error("Invalid Credentials")

def register():
    st.markdown("<h1>📝 Register</h1>", unsafe_allow_html=True)
    username = st.text_input("Create Username")
    password = st.text_input("Create Password", type="password")

    if st.button("Register"):
        hashed = hash_password(password)
        try:
            c.execute("INSERT INTO users VALUES(?,?)",
                      (username, hashed))
            conn.commit()
            st.success("Account Created Successfully!")
        except:
            st.error("Username Already Exists")

# -------------------------
# MAIN APP
# -------------------------
if not st.session_state.user:
    option = st.sidebar.selectbox("Select Option", ["Login", "Register"])
    if option == "Login":
        login()
    else:
        register()

else:

    st.sidebar.title(f"👋 {st.session_state.user}")
    page = st.sidebar.radio("Navigation", ["Dashboard", "Predict", "History"])
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.rerun()

    # ---------------- DASHBOARD ----------------
    if page == "Dashboard":
        st.markdown("<h1>📊 AI Analytics Dashboard</h1>", unsafe_allow_html=True)

        c.execute("SELECT fish, confidence FROM history WHERE username=?",
                  (st.session_state.user,))
        rows = c.fetchall()

        if not rows:
            st.info("No data available yet.")
        else:
            df = pd.DataFrame(rows, columns=["Fish", "Confidence"])

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Predictions", len(df))
            col2.metric("Average Confidence",
                        f"{round(df['Confidence'].mean(),2)}%")
            col3.metric("Unique Fish Types", df["Fish"].nunique())

            st.divider()

            st.subheader("Fish Distribution")
            st.bar_chart(df["Fish"].value_counts())

    # ---------------- PREDICT ----------------
    elif page == "Predict":
        st.markdown("<h1>🐟 Fish Detection</h1>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload Fish Image")

        if uploaded_file:
            st.image(uploaded_file, use_container_width=True)

            fish_classes = ["Salmon", "Tuna", "Tilapia",
                            "Catfish", "Carp"]

            prediction = random.choice(fish_classes)
            confidence = round(random.uniform(70, 99), 2)

            st.markdown(f"""
            <div class='glass'>
            <h2>Prediction: {prediction}</h2>
            <h3>Confidence: {confidence}%</h3>
            </div>
            """, unsafe_allow_html=True)

            c.execute("INSERT INTO history VALUES(?,?,?,?)",
                      (st.session_state.user,
                       prediction,
                       confidence,
                       datetime.datetime.now()))
            conn.commit()

    # ---------------- HISTORY ----------------
    elif page == "History":
        st.markdown("<h1>📜 Prediction History</h1>",
                    unsafe_allow_html=True)

        c.execute("SELECT fish, confidence, time FROM history WHERE username=?",
                  (st.session_state.user,))
        rows = c.fetchall()

        if not rows:
            st.info("No history available.")
        else:
            df = pd.DataFrame(rows,
                              columns=["Fish", "Confidence", "Time"])

            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode()
            st.download_button(
                "⬇ Download CSV",
                csv,
                "prediction_history.csv",
                "text/csv"
            )
