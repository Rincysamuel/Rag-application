import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

# -----------------------------
# Initialize Embeddings
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# Initialize Local LLM
# -----------------------------
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",   # ✅ lightweight local model
    tokenizer="google/flan-t5-small",
    max_length=512
)
llm = HuggingFacePipeline(pipeline=pipe)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="RAG PDF QA", page_icon="📄")
st.title("📄 Local PDF Question Answering (No API)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Save uploaded file locally
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and split PDF
    loader = PyPDFLoader("uploaded.pdf")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)

    # Create vectorstore
    vectorstore = FAISS.from_documents(splits, embeddings)

    # Initialize QA chain with memory
    if "qa" not in st.session_state:
        st.session_state.qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
        )
        st.session_state.chat_history = []

    # -----------------------------
    # Chat UI
    # -----------------------------
    question = st.text_input("Ask a question about the PDF:")

    if question:
        result = st.session_state.qa(
            {"question": question, "chat_history": st.session_state.chat_history}
        )
        st.session_state.chat_history.append((question, result["answer"]))

        # Display conversation
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}:** {q}")
            st.markdown(f"**A{i+1}:** {a}")
