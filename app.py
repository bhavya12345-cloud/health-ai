import streamlit as st
from langchain_ibm import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
import pandas as pd
import random
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="🩺 Health Assistant", layout="wide", page_icon="🩺")

# Initialize session state
if "current_section" not in st.session_state:
    st.session_state.current_section = "home"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "symptoms_history" not in st.session_state:
    st.session_state.symptoms_history = []
if "treatment_plan" not in st.session_state:
    st.session_state.treatment_plan = {}

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

# Top Navigation Buttons (Only 4 Major Sections)
st.markdown('<div style="display:flex; justify-content:center; gap:20px; margin-bottom:20px;">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🏠 Home", key="btn_home", use_container_width=True):
        st.session_state.current_section = "home"
with col2:
    if st.button("🧠 Symptoms", key="btn_symptoms", use_container_width=True):
        st.session_state.current_section = "symptoms"
with col3:
    if st.button("🤖 Chat", key="btn_chat", use_container_width=True):
        st.session_state.current_section = "chat"
with col4:
    if st.button("💊 Treatments", key="btn_treatments", use_container_width=True):
        st.session_state.current_section = "treatments"
st.markdown('</div>', unsafe_allow_html=True)

# Header
st.markdown('<h1 style="text-align:center; color:#2ecc71;">🩺 Health Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; font-size:16px;">A modern health tracking and wellness assistant.</p>', unsafe_allow_html=True)

# Function to render section content
def render_section(title, content):
    st.markdown(f'<div style="background-color:#fff; padding:20px; border-radius:10px; box-shadow:0 2px 4px rgba(0,0,0,0.1); margin:10px 0;">{title}</div>', unsafe_allow_html=True)
    st.markdown(content, unsafe_allow_html=True)

# ------------------------------ HOME PAGE ------------------------------
if st.session_state.current_section == "home":
    render_section(
        "<h2>🩺 Welcome to Your Personalized Health Assistant</h2>",
        """
        This application helps you manage your health comprehensively — from symptom checks to treatment planning.
        ### 🧠 Highlights:
        - 💬 AI-Powered Symptom Checker  
        - 🤖 AI Chatbot for advice  
        - 🎯 Customizable Treatment Plans  
        Get started by exploring any of the tools above!
        """
    )

# ------------------------------ SYMPTOM CHECKER ------------------------------
elif st.session_state.current_section == "symptoms":
    render_section("<h2>🧠 AI Symptom Checker</h2>", """
    <p>Analyze your symptoms and get possible conditions with likelihoods.</p>
    """)
    symptoms = st.text_area("Describe your symptoms:")
    if st.button("Check Symptoms"):
        with st.spinner("Analyzing..."):
            prompt = f"""
            Based on these symptoms: '{symptoms}', provide a list of possible conditions,
            their likelihood percentages, and next steps like when to see a doctor or self-care measures.
            Format the output as JSON.
            """
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

# ------------------------------ CHATBOT ------------------------------
elif st.session_state.current_section == "chat":
    render_section("<h2>🤖 AI Chatbot</h2>", """
    <p>Ask anything about health and get real-time responses from our AI assistant.</p>
    """)
    user_input = st.text_input("Ask anything about health...")
    if st.button("Send") and user_input:
        st.session_state.messages.append(("user", user_input))
        with st.spinner("Thinking..."):
            ai_response = llm.invoke(user_input)
            st.session_state.messages.append(("assistant", ai_response))

    for role, msg in st.session_state.messages:
        bubble_style = "background-color:#d6eaff;" if role == "user" else "background-color:#e6f0ff;"
        st.markdown(f'<div style="{bubble_style} padding:10px; border-radius:10px; max-width:70%; margin:5px auto;"><b>{role}:</b> {msg}</div>', unsafe_allow_html=True)

# ------------------------------ TREATMENTS ------------------------------
elif st.session_state.current_section == "treatments":
    render_section("<h2>💊 Personalized Treatment Planner</h2>", """
    <p>Enter a condition and patient details to generate an AI-powered treatment plan.</p>
    """)
    condition = st.text_input("Condition / Diagnosis")
    patient_details = st.text_area("Patient Details (Age, Gender, Comorbidities)")
    if st.button("Generate Treatment Plan"):
        with st.spinner("Generating plan..."):
            prompt = f"""
            Create a personalized treatment plan for a patient with:
            Condition: {condition}
            Details: {patient_details}
            Include medications, lifestyle changes, follow-up care, and duration.
            Format as JSON.
            """
            response = llm.invoke(prompt)
            try:
                plan = eval(response.strip())
                st.session_state.treatment_plan = plan
                st.json(plan)
            except Exception as e:
                st.error(f"Failed to parse treatment plan: {str(e)}")

# Footer
st.markdown("---")
st.markdown("© 2025 MyHospital Health Assistant | Built with ❤️ using Streamlit & Watsonx")

# Debug Mode
with st.expander("🔧 Debug Mode"):
    st.write("Session State:", st.session_state)
