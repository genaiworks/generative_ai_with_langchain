import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from utils import DocumentLoader
from doc_processor import configure_conversation


st.set_page_config(page_title="Chat Bot")
st.title(" Chat Bot")

uploaded_file = st.sidebar.file_uploader(
    label="Upload Files For Reference for Chat Bot",
    type=list(DocumentLoader.supported_extensions.keys()), 
    accept_multiple_files=True
)

if not uploaded_file:
    st.info("Upload Files For Reference for Chat Bot")
    st.stop()

qa_chain = configure_conversation(uploaded_file)
assistant = st.chat_message("assistant")
user_query = st.chat_input(placeholder="Ask any question")

if user_query:
    stream_handler = StreamlitCallbackHandler(assistant)
    response = qa_chain.run(user_query, callbacks=[stream_handler])
    st.markdown(response)