import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from utils.pdf_handler import RAGPipeline

st.set_page_config(page_title="PDFChat", layout="wide")
st.title("📄 PDFChat: AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_pipeline" not in st.session_state:
    with st.spinner("Loading AI Models..."):
        st.session_state.rag_pipeline = RAGPipeline()

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file and st.button("Ingest Document"):
        with st.spinner("Processing..."):
            status = st.session_state.rag_pipeline.process_pdf(uploaded_file)
            st.success(status)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        retriever = st.session_state.rag_pipeline.get_retriever()
        if not retriever:
            st.warning("⚠️ Upload a PDF first.")
        else:
            llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-3.5-turbo")
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "Answer based on context:\n\n{context}"),
                ("human", "{input}"),
            ])
            chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt_template))
            
            with st.spinner("Thinking..."):
                response = chain.invoke({"input": prompt})["answer"]
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})