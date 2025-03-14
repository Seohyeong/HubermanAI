import requests
import streamlit as st

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api.logger_config import logger

def get_api_response(question, chat_history, session_id):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {
        "question": question,
        "chat_history": chat_history[-1:]
    }
    if session_id:
        data["session_id"] = session_id

    try:
        response = requests.post("http://localhost:8000/chat", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"[Request] API request failed with status code {response.status_code}: {response.text}")
            st.error(f"[Request] API request failed with status code {response.status_code}: {response.text}")
            return None
    except Exception as e:
        logger.error(f"[Request] An error occurred: {str(e)}")
        st.error(f"[Request] An error occurred: {str(e)}")
        return None