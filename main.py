import streamlit as st
import base64
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from utils.pdf_handler import RAGPipeline

st.set_page_config(page_title="PDFChat", layout="wide")
st.title("📄 PDFChat: Side-by-Side Assistant")

st.markdown("""
**To begin follow these easy steps:**
1. Upload a PDF (sidebar)
2. Click **"Ingest Document"**
3. Begin chatting!
""")

# pdf viewer
def display_pdf(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_pipeline" not in st.session_state:
    with st.spinner("Loading AI Models..."):
        st.session_state.rag_pipeline = RAGPipeline()

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file and st.button("Ingest Document"):
        with st.spinner("Processing PDF (Chunking & Embedding)..."):
            status = st.session_state.rag_pipeline.process_pdf(uploaded_file)
            st.success(status)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("💬 Chat")
    
    chat_container = st.container(height=600)
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    #chat input
    if prompt := st.chat_input("Ask a question about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container.chat_message("user"):
            st.markdown(prompt)

        with chat_container.chat_message("assistant"):
            retriever = st.session_state.rag_pipeline.get_retriever()
            
            if not retriever:
                st.warning("⚠️ Please upload and ingest a document first.")
            else:
                llm = ChatOpenAI(
                    api_key=st.secrets["OPENAI_API_KEY"], 
                    model="gpt-3.5-turbo",
                    temperature=0
                )
                
                system_prompt = (
                "You are an expert academic professor. Your goal is to help a student "
                "understand the material deeply, not just give short answers. "
                "You are provided with snippets from a textbook or paper below. "
                "\n\n"
                "Guidelines for your response:"
                "\n1. **Be Pedagogical:** Explain concepts clearly. If a term is complex, define it first."
                "\n2. **Use Structure:** Use bullet points, numbered lists, or bold text to break down long explanations."
                "\n3. **Cite Evidence:** Explicitly reference the text (e.g., 'According to the document...') to back up your claims."
                "\n4. **Admit Gaps:** If the answer is not in the provided context, state clearly: "
                "'The provided text does not contain information about this.' Do not make things up."
                "\n\n"
                "--- TEXTBOOK CONTEXT START ---"
                "\n{context}"
                "\n--- TEXTBOOK CONTEXT END ---"
                )
                
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])

                question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                with st.spinner("Thinking..."):
                    response = rag_chain.invoke({"input": prompt})
                    st.markdown(response["answer"])
                    
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

with col2:
    st.subheader("📄 Document Viewer")
    if uploaded_file:
        display_pdf(uploaded_file)
    else:
        st.info("Upload a PDF to see it here.")