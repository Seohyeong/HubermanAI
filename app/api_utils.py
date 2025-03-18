import requests
import streamlit as st


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
        response = requests.post(st.secrets["API_GATEWAY_ENDPOINT"], headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"[Request] API request failed with status code {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"[Request] An error occurred: {str(e)}")
        return None