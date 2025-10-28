import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
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
    model="google/flan-t5-base",     # ✅ better than small, still runs locally
    tokenizer="google/flan-t5-base",
    max_length=512
)
llm = HuggingFacePipeline(pipeline=pipe)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="RAG PDF QA", page_icon="📄")
st.title("📄 Local PDF Question Answering (No API)")

uploaded_file = st.file_uploader("📂 Upload a PDF file", type=["pdf"])

# Reset chat when a new PDF is uploaded
if uploaded_file:
    if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.qa = None
        st.session_state.chat_history = []
        st.session_state.last_uploaded_file = uploaded_file.name

    # Save uploaded file locally
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # -----------------------------
    # Load and split PDF
    # -----------------------------
    loader = PyPDFLoader("uploaded.pdf")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)

    # Create vectorstore
    vectorstore = FAISS.from_documents(splits, embeddings)

    # -----------------------------
    # Custom Prompt
    # -----------------------------
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an AI assistant that answers questions only based on the provided PDF content.\n"
            "If the answer is not found in the context, say 'I could not find that information in the document.'\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer clearly and concisely:"
        ),
    )

    # -----------------------------
    # Initialize QA Chain
    # -----------------------------
    if "qa" not in st.session_state or st.session_state.qa is None:
        st.session_state.qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            combine_docs_chain_kwargs={"prompt": prompt},
        )
        st.session_state.chat_history = []

    # -----------------------------
    # Chat Interface
    # -----------------------------
    st.markdown("### 💬 Ask questions about your PDF")

    question = st.text_input("Your Question:")
    if st.button("Get Answer") and question:
        result = st.session_state.qa(
            {"question": question, "chat_history": st.session_state.chat_history}
        )
        st.session_state.chat_history.append((question, result["answer"]))

    # Display conversation history
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.markdown(f"**Q{i+1}:** {q}")
        st.markdown(f"**A{i+1}:** {a}")
