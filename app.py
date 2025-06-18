import streamlit as st
from langchain_ibm import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
import pandas as pd
import random
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="🩺 Health Assistant", layout="wide", page_icon="🩺")

# Custom CSS for animated UI and green/blue theme
st.markdown("""
    <style>
        body {
            background-color: #f0fff4;
            font-family: Arial, sans-serif;
        }
        .main {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            transition: all 0.3s ease-in-out;
        }
        .card {
            background-color: #ffffff;
            padding: 15px 20px;
            border-left: 5px solid #2ecc71;
            margin: 10px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            animation: fadeIn 0.5s ease-in-out;
        }
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(10px);}
            to {opacity: 1; transform: translateY(0);}
        }
        .chat-bubble-user {
            background-color: #d6eaff;
            padding: 10px;
            border-radius: 10px;
            max-width: 70%;
            align-self: flex-end;
            margin: 5px 0;
        }
        .chat-bubble-bot {
            background-color: #e6f0ff;
            padding: 10px;
            border-radius: 10px;
            max-width: 70%;
            align-self: flex-start;
            margin: 5px 0;
        }
        .navbar {
            display: flex;
            justify-content: center;
            gap: 15px;
            padding: 10px 0;
            background: linear-gradient(to right, #2ecc71, #27ae60);
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .nav-button {
            background-color: #ffffff;
            color: #2ecc71;
            border: none;
            padding: 10px 16px;
            font-size: 16px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .nav-button:hover {
            background-color: #eafaf1;
        }
        .fade-enter {
            opacity: 0;
            transform: translateY(10px);
        }
        .fade-enter-active {
            opacity: 1;
            transform: translateY(0);
            transition: all 0.3s ease;
        }
        .metric-box {
            padding: 10px;
            border-radius: 8px;
            background-color: #ecf0f1;
            margin: 5px;
            text-align: center;
        }
        .positive {
            color: green;
        }
        .negative {
            color: red;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "current_section" not in st.session_state:
    st.session_state.current_section = "home"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "symptoms_history" not in st.session_state:
    st.session_state.symptoms_history = []
if "treatment_plan" not in st.session_state:
    st.session_state.treatment_plan = {}
if "profile" not in st.session_state:
    st.session_state.profile = {}
if "generated_data" not in st.session_state:
    st.session_state.generated_data = []

# Load Watsonx credentials from secrets
try:
    credentials = {
        "url": st.secrets["WATSONX_URL"],
        "apikey": st.secrets["WATSONX_APIKEY"]
    }
    project_id = st.secrets["WATSONX_PROJECT_ID"]
    llm = WatsonxLLM(
        model_id="ibm/granite-3-2-8b-instruct",
        url=credentials.get("url"),
        apikey=credentials.get("apikey"),
        project_id=project_id,
        params={
            GenParams.DECODING_METHOD: "greedy",
            GenParams.TEMPERATURE: 0.7,
            GenParams.MIN_NEW_TOKENS: 5,
            GenParams.MAX_NEW_TOKENS: 500,
            GenParams.STOP_SEQUENCES: ["Human:", "Observation"],
        },
    )
except KeyError:
    st.warning("⚠️ Watsonx credentials missing.")
    st.stop()
except Exception as e:
    st.error(f"🚨 Error initializing LLM: {str(e)}")
    st.stop()

# Top Navigation Buttons
st.markdown('<div class="navbar">', unsafe_allow_html=True)
col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
with col1:
    if st.button("🏠 Home", key="btn_home", use_container_width=True):
        st.session_state.current_section = "home"
with col2:
    if st.button("🔐 Login", key="btn_login", use_container_width=True):
        st.session_state.current_section = "login"
with col3:
    if st.button("🧾 Profile", key="btn_profile", use_container_width=True):
        st.session_state.current_section = "profile"
with col4:
    if st.button("🧠 Symptoms", key="btn_symptoms", use_container_width=True):
        st.session_state.current_section = "symptoms"
with col5:
    if st.button("🤖 Chat", key="btn_chat", use_container_width=True):
        st.session_state.current_section = "chat"
with col6:
    if st.button("🫀 Diseases", key="btn_diseases", use_container_width=True):
        st.session_state.current_section = "diseases"
with col7:
    if st.button("📈 Reports", key="btn_reports", use_container_width=True):
        st.session_state.current_section = "reports"
with col8:
    if st.button("💊 Treatments", key="btn_treatments", use_container_width=True):
        st.session_state.current_section = "treatments"
with col9:
    if st.button("⚙️ Settings", key="btn_settings", use_container_width=True):
        st.session_state.current_section = "settings"
st.markdown('</div>', unsafe_allow_html=True)

# Header
st.markdown('<h1 style="text-align:center; color:#2ecc71;">🩺 Health Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; font-size:16px;">A modern health tracking and wellness assistant.</p>', unsafe_allow_html=True)

# Function to show/hide sections with animation
def render_section(title, content):
    st.markdown(f'<div class="card fade-enter-active">{title}</div>', unsafe_allow_html=True)
    st.markdown(content, unsafe_allow_html=True)

# ------------------------------ HOME PAGE ------------------------------
if st.session_state.current_section == "home":
    render_section(
        "<h2>🩺 Welcome to Your Personalized Health Assistant</h2>",
        """
        This application helps you manage your health comprehensively — from symptom checks to fitness planning.
        ### 🧠 Highlights:
        - 💬 AI-Powered Symptom Checker  
        - 📊 Real-Time Health Metrics  
        - 🎯 Customizable Wellness Plans  
        - 🤖 AI Chatbot for advice  
        - 📈 Weekly Reports powered by AI  
        Get started by exploring any of the tools above!
        """
    )

# ------------------------------ LOGIN PAGE ------------------------------
elif st.session_state.current_section == "login":
    render_section("<h2>🔐 Login</h2>", """
        <form>
            <label>Username:</label><br>
            <input type="text" placeholder="Enter username"><br><br>
            <label>Password:</label><br>
            <input type="password" placeholder="Enter password"><br><br>
            <button>Login</button>
        </form>
    """)

# ------------------------------ USER PROFILE ------------------------------
elif st.session_state.current_section == "profile":
    st.markdown('<div class="card fade-enter-active">', unsafe_allow_html=True)
    st.markdown('<h2>🧾 User Profile & Dashboard</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name")
        age = st.number_input("Age", min_value=0, max_value=120)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    with col2:
        height = st.number_input("Height (cm)", min_value=50, max_value=250)
        weight = st.number_input("Weight (kg)", min_value=10, max_value=300)
        if height > 0:
            bmi = weight / ((height / 100) ** 2)
            st.write(f"**BMI:** {bmi:.1f}")
    if st.button("Save Profile"):
        st.session_state.profile = {"name": name, "age": age, "gender": gender, "height": height, "weight": weight}
        prompt = f"Give general health tips for a {age}-year-old {gender} with height {height} cm and weight {weight} kg."
        response = llm.invoke(prompt)
        st.markdown(f"💡 **AI Tip:** {response}")
        st.success("Profile saved!")
    st.markdown('</div>')

# ------------------------------ SYMPTOM CHECKER ------------------------------
elif st.session_state.current_section == "symptoms":
    st.markdown('<div class="card fade-enter-active">', unsafe_allow_html=True)
    st.markdown('<h2>🧠 AI Symptom Checker</h2>', unsafe_allow_html=True)
    symptoms = st.text_area("Describe your symptoms:")
    if st.button("Check Symptoms"):
        with st.spinner("Analyzing..."):
            prompt = """
            Based on these symptoms: '{symptoms}', provide a list of possible conditions,
            their likelihood percentages, and next steps like when to see a doctor or self-care measures.
            Format the output as JSON.
            """.format(symptoms=symptoms)
            response = llm.invoke(prompt)
            try:
                result = eval(response.strip())  # assuming structured format
                st.session_state.symptoms_history.append({"input": symptoms, "response": result})
                st.json(result)
            except Exception as e:
                st.error(f"Invalid response format from AI: {str(e)}")
    st.markdown("### 📜 Symptom History")
    for item in st.session_state.symptoms_history:
        st.markdown(f"**Q:** {item['input']}")
        st.json(item['response'])
        st.divider()
    st.markdown('</div>')

# ------------------------------ CHATBOT ------------------------------
elif st.session_state.current_section == "chat":
    st.markdown('<div class="card fade-enter-active">', unsafe_allow_html=True)
    st.markdown('<h2>🤖 AI Chatbot</h2>', unsafe_allow_html=True)
    user_input = st.text_input("Ask anything about health...")
    if st.button("Send") and user_input:
        st.session_state.messages.append(("user", user_input))
        with st.spinner("Thinking..."):
            ai_response = llm.invoke(user_input)
            st.session_state.messages.append(("assistant", ai_response))
    for role, msg in st.session_state.messages:
        bubble_class = "chat-bubble-user" if role == "user" else "chat-bubble-bot"
        st.markdown(f'<div class="{bubble_class}"><b>{role}:</b> {msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>')

# ------------------------------ TREATMENTS ------------------------------
elif st.session_state.current_section == "treatments":
    st.markdown('<div class="card fade-enter-active">', unsafe_allow_html=True)
    st.markdown('<h2>💊 Personalized Treatment Planner</h2>', unsafe_allow_html=True)
    condition = st.text_input("Condition / Diagnosis")
    patient_details = st.text_area("Patient Details (Age, Gender, Comorbidities)")
    if st.button("Generate Treatment Plan"):
        with st.spinner("Generating plan..."):
            prompt = """
            Create a personalized treatment plan for a patient with:
            Condition: {condition}
            Details: {patient_details}
            Include medications, lifestyle changes, follow-up care, and duration.
            Format as JSON.
            """.format(condition=condition, patient_details=patient_details)
            response = llm.invoke(prompt)
            try:
                plan = eval(response.strip())
                st.session_state.treatment_plan = plan
                st.json(plan)
            except Exception as e:
                st.error(f"Failed to parse treatment plan: {str(e)}")
    st.markdown('</div>')

# ------------------------------ REPORTS ------------------------------
elif st.session_state.current_section == "reports":
    st.markdown('<div class="card fade-enter-active">', unsafe_allow_html=True)
    st.markdown('<h2>📈 Progress Reports</h2>', unsafe_allow_html=True)
    days = st.slider("Days of Trend", 1, 30, value=7)
    dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
    heart_rates = [random.randint(60, 100) for _ in range(days)]
    glucose_levels = [round(random.uniform(70, 140), 1) for _ in range(days)]
    blood_pressure = [(random.randint(110, 130), random.randint(70, 90)) for _ in range(days)]
    df = pd.DataFrame({
        "Date": dates,
        "Heart Rate": heart_rates,
        "Glucose Level": glucose_levels,
        "Systolic BP": [bp[0] for bp in blood_pressure],
        "Diastolic BP": [bp[1] for bp in blood_pressure]
    })
    st.line_chart(df.set_index("Date")[["Heart Rate", "Glucose Level"]])
    st.line_chart(df.set_index("Date")[["Systolic BP", "Diastolic BP"]])
    st.markdown("### Metric Summary")
    avg_hr = round(sum(heart_rates) / len(heart_rates))
    avg_gluc = round(sum(glucose_levels) / len(glucose_levels))
    st.markdown(f"<div class='metric-box'>Avg Heart Rate: {avg_hr} bpm <span class='positive'>▲+1</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-box'>Avg Glucose: {avg_gluc} mg/dL <span class='negative'>▼-2</span></div>", unsafe_allow_html=True)
    if st.button("Generate AI Report Summary"):
        prompt = f"Provide insights based on these health trends: {df.describe().to_string()}. Give actionable advice."
        summary = llm.invoke(prompt)
        st.markdown(f"📊 **AI Analysis:**\n{summary}")
    st.markdown('</div>')

# ------------------------------ CHRONIC DISEASE MANAGEMENT ------------------------------
elif st.session_state.current_section == "diseases":
    st.markdown('<div class="card fade-enter-active">', unsafe_allow_html=True)
    st.markdown('<h2>🫀 Chronic Disease Logs</h2>', unsafe_allow_html=True)
    condition = st.selectbox("Condition", ["Diabetes", "Hypertension", "Asthma"])
    if condition == "Diabetes":
        st.markdown("### 🩸 Blood Glucose Tracker")
        glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=40, max_value=400, step=5)
        if st.button("Log Glucose"):
            st.session_state.glucose_log = st.session_state.get("glucose_log", []) + [glucose]
            st.success(f"Logged: {glucose} mg/dL")
            prompt = f"My blood sugar is {glucose}. Is it normal? What should I do?"
            try:
                advice = llm.invoke(prompt)
            except:
                advice = "AI is currently unavailable for advice."
            st.markdown(f"🤖 **AI Advice:**\n{advice}")
        if "glucose_log" in st.session_state and len(st.session_state.glucose_log) > 0:
            df_glucose = pd.DataFrame({
                "Date": [datetime.now() - timedelta(days=i) for i in range(len(st.session_state.glucose_log))],
                "Glucose Level (mg/dL)": st.session_state.glucose_log
            })
            st.line_chart(df_glucose.set_index("Date")["Glucose Level (mg/dL)"])
    elif condition == "Hypertension":
        st.markdown("### 💓 Blood Pressure Log")
        col1, col2 = st.columns(2)
        with col1:
            systolic = st.number_input("Systolic (mmHg)", min_value=90, max_value=200, value=120)
        with col2:
            diastolic = st.number_input("Diastolic (mmHg)", min_value=60, max_value=130, value=80)
        if st.button("Log BP"):
            st.session_state.bp_log = st.session_state.get("bp_log", []) + [(systolic, diastolic)]
            st.success(f"Logged: {systolic}/{diastolic} mmHg")
            prompt = f"My blood pressure is {systolic}/{diastolic} mmHg. What does that mean?"
            try:
                advice = llm.invoke(prompt)
            except:
                advice = "AI is currently unavailable for advice."
            st.markdown(f"🤖 **AI Advice:**\n{advice}")
        if "bp_log" in st.session_state and len(st.session_state.bp_log) > 0:
            bp_data = pd.DataFrame(st.session_state.bp_log, columns=["Systolic", "Diastolic"])
            bp_data["Date"] = [datetime.now() - timedelta(days=i) for i in range(len(bp_data))]
            st.line_chart(bp_data.set_index("Date")[["Systolic", "Diastolic"]])
    elif condition == "Asthma":
        st.markdown("### 🌬️ Asthma Trigger Tracker")
        triggers = st.text_area("Triggers Today (e.g., pollen, dust)")
        severity = st.slider("Severity (1-10)", 1, 10)
        if st.button("Log Asthma Episode"):
            st.session_state.asthma_log = st.session_state.get("asthma_log", []) + [{"triggers": triggers, "severity": severity}]
            st.success("Episode logged successfully.")
            prompt = f"What are some ways to avoid asthma triggers like {triggers}?"
            try:
                advice = llm.invoke(prompt)
            except:
                advice = "AI is currently unavailable for advice."
            st.markdown(f"🤖 **AI Advice:**\n{advice}")
        if "asthma_log" in st.session_state and len(st.session_state.asthma_log) > 0:
            asthma_df = pd.DataFrame(st.session_state.asthma_log)
            asthma_df["Date"] = [datetime.now() - timedelta(days=i) for i in range(len(asthma_df))]
            st.line_chart(asthma_df.set_index("Date")["severity"])
    st.markdown('</div>')

# Footer
st.markdown("---")
st.markdown("© 2025 MyHospital Health Assistant | Built with ❤️ using Streamlit & Watsonx")

# Debug Mode
with st.expander("🔧 Debug Mode"):
    st.write("Session State:", st.session_state)
