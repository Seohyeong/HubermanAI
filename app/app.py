import streamlit as st
from api_utils import get_api_response


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
                           
                
st.title("Huberman RAG Chatbot")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None


# Display the chat interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Query:"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧐"):
        st.markdown(prompt)

    with st.spinner("Generating response..."):
        response = get_api_response(question = prompt, 
                                    chat_history = st.session_state.chat_history, 
                                    session_id = st.session_state.session_id)
        
        if response:
            st.session_state.session_id = response.get('session_id')
            st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})
            
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(response['answer'])
                
                if response["docs"]:
                    st.write("Check out more information in following segments:")
                    num_videos = len(response["docs"])
                    
                    for i in range(num_videos):
                        col_video, col_meta = st.columns([2, 1.5])
                        
                        video_title = response["docs"][i]["title"]
                        video_header = response["docs"][i]["header"]
                        video_start = response["docs"][i]["time_start"]
                        video_end = response["docs"][i]["time_end"]
                        f_video_start, f_video_end = format_timestamp(video_start), format_timestamp(video_end)
                        video_url = "https://www.youtube.com/{}".format(
                            response["docs"][i]["video_id"]
                        )
                        video_url_start_end = "https://www.youtube.com/{}?start={}&end={}".format(
                            response["docs"][i]["video_id"],
                            str(convert_to_sec(video_start)),
                            str(convert_to_sec(video_end))
                            )
                            
                        with col_video:
                            st.video(video_url, start_time=f_video_start, end_time=f_video_end)
                        with col_meta:
                            st.markdown("""
                                        **{}**  
                                        *{}*
                                        """.format(video_title, video_header))
                            st.caption(video_url_start_end) 
        else:
            st.error("Failed to get a response from the API. Please try again.")