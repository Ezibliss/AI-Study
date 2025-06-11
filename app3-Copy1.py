import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI 
from langchain.chains import RetrievalQA
import os
import fitz  # PyMuPDF
import docx
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_xai import ChatXAI
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document

api_key = st.secrets["xai_api_key"]

# Upload the document
uploaded_file = st.file_uploader("📄 Upload a PDF or DOCX file", type=["pdf", "docx"])

if uploaded_file:
    st.success("✅ File uploaded successfully!")

    text = ""

    if uploaded_file.name.endswith(".pdf"):
        try:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in doc:
                try:
                    text += page.get_text()
                except Exception as e:
                    st.warning(f"⚠️ Skipping a page due to error: {e}")
        except Exception as e:
            st.error(f"❌ Failed to read PDF: {e}")
            text = ""
    
    elif uploaded_file.name.endswith(".docx"):
        try:
            doc = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            st.error(f"❌ Failed to read DOCX: {e}")
            text = ""
    else:
        st.warning("⚠️ Unsupported file type")
        text = ""

    if text:
        st.text_area("📄 Extracted Text", text[:1000])


        
    # Ask for XAI API Key
    #xai_api_key = st.text_input("xai_api_key")

    #if xai_api_key:
        #os.environ["xai_api_key"] = xai_api_key  # Optional, if the SDK uses env variable

        # Step 1: Split into chunks
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]

        # Step 2: Embed chunks
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
        db = FAISS.from_documents(docs, embeddings)

        # Step 3: Connect to XAI LLM
        # If LangChain supports XAI via ChatOpenAI-compatible wrapper:
        llm = ChatXAI(
            temperature=0.3,
            api_key=xai_api_key,
            openai_api_base="https://api.x.ai/v1",   # Replace with actual XAI base URL
            model="grok-3-mini-fast"  # Replace with your actual model name
        )

        # Step 4: Retrieval-based QA
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=db.as_retriever()
        )

        # Step 5: Ask a question
        user_question = st.text_input("💬 Ask a question about your course:")
        if user_question:
            answer = qa_chain.run(user_question)
            st.markdown("🎓 *Answer:* " + answer)

import streamlit as st

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #f0f4f8;
            background-image: url("AI Study.jpg");
            background-size: cover;
            background-position: center;
            padding: 2rem;
            color: #fff;
        }

        h1, h2, h3 {
            color: #ffffff !important;
        }

        .stTextInput > div > div > input {
            background-color: #ffffff;
            color: #000;
        }

        .stTextArea > div > textarea {
            background-color: #ffffff;
            color: #000;
        }

        .css-1v0mbdj {
            background-color: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 15px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 Your Study Companion")

col1, col2 = st.columns([1, 2])
with col1:
    st.image("Studylogo.png", width=100)
with col2:
    st.title("Your Study Companion")

