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
import base64



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
    xai_api_key = st.text_input("xai_api_key")

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
        api_key = st.secrets["xai_api_key"]
        
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



# Load local background image and convert to base64
with open("AI Study.jpg", "rb") as image_file:  # Replace with your file name
    img_bytes = image_file.read()
    img_base64 = base64.b64encode(img_bytes).decode()

# Inject custom CSS using local image
st.markdown(f"""
    <style>
        .stApp {{
            background-image: url("data:image/png;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        .css-1v0mbdj {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 20px;
            border-radius: 15px;
        }}

        h1, h2, h3 {{
            color: #ffffff !important;
        }}

        .stTextInput > div > div > input,
        .stTextArea > div > textarea {{
            background-color: #ffffff;
            color: #000000;
        }}
    </style>
""", unsafe_allow_html=True)

# Display logo and title
col1, col2 = st.columns([1, 3])
with col1:
    st.image("Studylogo.png", width=120)  # Replace with your logo file
with col2:
    st.markdown("Your Study Companion")

st.markdown("Welcome! Please Upload your materials and ask any question.")
