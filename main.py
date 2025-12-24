import streamlit as st
import time # <--- Added for animation
import base64
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from streamlit_pdf_viewer import pdf_viewer
from utils.pdf_handler import RAGPipeline

st.set_page_config(page_title="PDFChat", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()

def stream_text(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

def submit():
    user_input = st.session_state.widget
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.widget = ""

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file:
        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                status = st.session_state.rag_pipeline.process_pdf(uploaded_file)
                st.session_state.last_uploaded = uploaded_file.name
                st.success(status)

if not uploaded_file:
    st.title("📄 PDFChat: AI Assistant")
    st.markdown("""
    **To begin follow these easy steps:**
    1. Upload a PDF (sidebar)
    2. Wait for the green "Success" message
    3. Begin chatting!
    """)

else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💬 Chat")
        
        chat_container = st.container(height=650, border=True)
        
        with chat_container:
            if not st.session_state.messages:
                 st.info("👋 Ask a question about the document to begin.")
            
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        st.text_input(
            "Ask a question...", 
            key="widget", 
            on_change=submit,
            label_visibility="collapsed",
            placeholder="Type your question here..."
        )

    with col2:
        st.subheader("📄 Viewer")
        
        with st.container(height=710, border=True):
            binary_data = uploaded_file.getvalue()
            pdf_viewer(input=binary_data, height=700)

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        
        with chat_container.chat_message("assistant"):
            retriever = st.session_state.rag_pipeline.get_retriever()
            
            if retriever:
                retriever.search_kwargs['k'] = 10
            
            if not retriever:
                st.error("⚠️ Pipeline not ready. Please re-upload the document.")
            else:
                llm = ChatOpenAI(
                    api_key=st.secrets["OPENAI_API_KEY"], 
                    model="gpt-4o",
                    temperature=0
                )
                
                system_prompt = (
                    "You are an expert academic professor. "
                    "Use the provided context to answer the student's question. "
                    "\n\n"
                    "Rules:"
                    "\n1. If the answer is not explicitly in the context, say 'I cannot find that specific information in the retrieved text,' but try to summarize what IS available."
                    "\n2. Ignore headers and footers, but use the main text content."
                    "\n3. Always cite the text."
                    "\n\n"
                    "--- CONTEXT START ---"
                    "\n{context}"
                    "\n--- CONTEXT END ---"
                )
                
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])

                question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                with st.spinner("Analyzing text..."):
                    try:
                        last_question = st.session_state.messages[-1]["content"]
                        response = rag_chain.invoke({"input": last_question})
                        
                        full_response = response["answer"]
                        st.write_stream(stream_text(full_response))
                        
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        
                    except Exception as e:
                        st.error(f"Error generating response: {e}")