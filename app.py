import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama

# 1. UI Page Configuration Layout
st.set_page_config(page_title="Private Doc Intel", page_icon="🔒", layout="wide")
st.title("🔒 Private Document Intelligence Platform")
st.subheader("Query your private PDFs completely offline using local hardware acceleration.")

MODEL_NAME = "llama3.2:1b"
DB_DIR = "./chroma_db"

# Ensure database directory path exists
if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

# 2. Left Sidebar: Document Ingestion Layer
with st.sidebar:
    st.header("📄 Data Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF document:", type=["pdf"])
    
    if uploaded_file is not None:
        # Save temporary file locally to allow LangChain parsing loaders to read it
        temp_path = os.path.join("./", uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.info("Parsing document contents...")
        
        # Load and slice data chunks
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        
        st.info(f"Generated {len(chunks)} contextual vector matrices. Indexing database...")
              
        # Generate Vector Embeddings natively using the dedicated embedding model
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)

        
        
        st.success("Database fully indexed! You can now ask questions.")
        
        # Clean up the temporary structural file asset
        os.remove(temp_path)

# 3. Main Chat State Infrastructure Management
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! Upload a document on the left, and I will strictly use its context to answer your questions."}
    ]

# Display ongoing streaming message layouts
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. User Interaction Processing Engine
if user_prompt := st.chat_input("Ask something about your document..."):
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    
    # Query database for localized ground truth matching chunks
    context_data = ""
try:
    # Update this line to use the correct model too!
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(user_prompt)
    context_data = "\n\n".join([doc.page_content for doc in relevant_docs])
except Exception as e:
    context_data = "No matching document segments indexed yet."

    # Build advanced context augmented framework prompt payload
    system_rules = (
        "You are a strict technical compliance assistant. Answer the question using ONLY the provided text block below. "
        "If the answer cannot be found in the context, say 'I cannot find that information in the uploaded document.'\n\n"
        f"### DOCUMENT CONTEXT DATA:\n{context_data}"
    )
    
    # Compile messages mapping structure
    input_messages = [{"role": "system", "content": system_rules}]
    for msg in st.session_state.chat_history:
        input_messages.append({"role": msg["role"], "content": msg["content"]})
        
    # Render streaming output blocks directly into UI window element
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_text_output = ""
        
        try:
            stream = ollama.chat(model=MODEL_NAME, messages=input_messages, stream=True)
            for chunk in stream:
                text_chunk = chunk['message']['content']
                full_text_output += text_chunk
                response_placeholder.markdown(full_text_output + "▌")
            response_placeholder.markdown(full_text_output)
            st.session_state.chat_history.append({"role": "assistant", "content": full_text_output})
        except Exception as e:
            st.error(f"Execution boundary error: {e}")
