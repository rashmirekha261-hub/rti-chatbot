import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Setup Streamlit Page Config
st.set_page_config(page_title="RTI Act Chatbot", page_icon="⚖️", layout="centered")
st.title("⚖️ RTI Act Handbook Chatbot")
st.write("Ask questions about the Right to Information Act Handbook for Public Authorities.")

# 2. Get API Key from Environment Variables
if "OPENAI_API_KEY" not in os.environ:
    st.info("Please add your OpenAI API key to continue.", icon="🔑")
    openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        st.rerun()
    st.stop()

# 3. Load and Process the PDF (Cached so it only runs once)
@st.cache_resource
def initialize_vector_store():
    pdf_url = "https://www.mcrhrdi.gov.in/download/publications/RTI%20NEW%20BOOKS/2.The%20Right%20to%20information%20Act-A%20Handbook%20for%20Public%20Authorities-2021%20Ed.pdf"
    
    with st.spinner("Downloading and processing the RTI Handbook... This may take a minute."):
        loader = PyPDFLoader(pdf_url)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        
        return vectorstore.as_retriever(search_kwargs={"k": 5})

retriever = initialize_vector_store()

# 4. Setup LLM and RAG Pipeline
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

system_prompt = (
    "You are an expert legal assistant specializing in India's Right to Information (RTI) Act.\n"
    "Answer the user's question using strictly the provided context below. "
    "If you do not know the answer or if it's not mentioned in the context, say exactly "
    "'I cannot find that information in the provided RTI handbook.' Do not make things up.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
])

# Helper function to format retrieved documents into a text block
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Combine elements directly using the modern | operator
rag_pipeline = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Build the Chat History UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("e.g., What is the time limit for disposing an RTI request?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        with st.spinner("Searching handbook..."):
            answer = rag_pipeline.invoke(user_query)
            st.markdown(answer)
            
    st.session_state.messages.append({"role": "assistant", "content": answer})
