import streamlit as st
from api_utils import get_api_response

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api.logger_config import logger

def format_timestamp(timestamp: str) -> str:
    ts = timestamp.strip().split(":")
    if len(ts) == 3:
        return "{}h{}m{}s".format(ts[0], ts[1], ts[2])
    elif len(ts) == 2:
        return "{}m{}s".format(ts[0], ts[1])
    elif len(ts) == 1:
        return "{}s".format(ts[0])
    else:
        return None
    
def convert_to_sec(timestamp: str) -> int:
    ts = timestamp.strip().split(":")
    if len(ts) == 3:
        return int(ts[0]) * 3600 + int(ts[1]) * 60 + int(ts[2])
    elif len(ts) == 2:
        return int(ts[0]) * 60 + int(ts[1])
    elif len(ts) == 1:
        return int(ts[0])
    else:
        return None                       

def create_flattened_chat_history(chat_history):
    flattened_chat_history = []
    for pair in chat_history:
        user_msg, assistant_msg = pair
        if user_msg["is_valid"]:
            flattened_chat_history.append({"role": user_msg["role"], "content": user_msg["content"]})
            flattened_chat_history.append({"role": assistant_msg["role"], "content": assistant_msg["content"]})
    return flattened_chat_history

def display_question(question):
    st.markdown("""#### {}""".format(question))
    
def display_answer(answer):
    answer = answer.replace("#", "")
    st.markdown(answer)
    
def display_expander(videos):
    with st.expander("Sources and related videos"):
        for video in videos:
            col_video, col_meta = st.columns([1.5, 2])
            
            video_title = video["title"]
            video_header = video["header"]
            video_start = video["time_start"]
            video_end = video["time_end"]
            f_video_start, f_video_end = format_timestamp(video_start), format_timestamp(video_end)
            video_url = "https://www.youtube.com/watch?v={}".format(
                video["video_id"])
            video_url_start_end = "https://www.youtube.com/watch?v={}?start={}&end={}".format(
                video["video_id"],
                str(convert_to_sec(video_start)),
                str(convert_to_sec(video_end)))
  
            with col_video:
                st.video(video_url, start_time=f_video_start, end_time=f_video_end)
            with col_meta:
                st.markdown("""
                            **{}** [\[Watch on Youtube\]]({})  
                            :gray[*{}*]
                            """.format(video_title, video_url_start_end, video_header))
     

logger.info("Starting Streamlit application")
st.markdown("""
    <style>
        [data-testid="stDecoration"] {
            display: none;
        }
    </style>""",
    unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Ask Huberman Lab</h2>", unsafe_allow_html=True)


# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None
    
if "videos" not in st.session_state:
    st.session_state.videos = []

                  
for messages, videos in zip(st.session_state.chat_history, st.session_state.videos):
    display_question(messages[0]["content"])
    display_answer(messages[1]["content"])  
    display_expander(videos)


if prompt := st.chat_input("Ask anything"):
    display_question(prompt)

    with st.spinner("Generating response..."):
        chat_history = create_flattened_chat_history(st.session_state.chat_history)
        response = get_api_response(question = prompt, 
                                    chat_history = chat_history, 
                                    session_id = st.session_state.session_id)
            
        if response:
            st.session_state.session_id = response.get("session_id")  
            st.session_state.chat_history.append([{"role": "user", "content": prompt, "is_valid": response.get("is_valid")},
                                                {"role": "assistant", "content": response["answer"], "is_valid": response.get("is_valid")}])
                
            display_answer(response["answer"])

            if response["docs"]:
                videos = response["docs"]
                display_expander(videos) 
                st.session_state.videos.append(videos)
            else:
                st.session_state.videos.append([])
                
        else:
            logger.error("[Streamlit] Failed to get a response from the API.")
            st.error("Failed to get a response from the API. Please try again.")