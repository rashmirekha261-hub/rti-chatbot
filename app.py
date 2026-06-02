import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

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
    # This is the exact link to the handbook you provided
    pdf_url = "https://www.mcrhrdi.gov.in/download/publications/RTI%20NEW%20BOOKS/2.The%20Right%20to%20information%20Act-A%20Handbook%20for%20Public%20Authorities-2021%20Ed.pdf"
    
    with st.spinner("Downloading and processing the RTI Handbook... This may take a minute."):
        # Load PDF directly from the URL
        loader = PyPDFLoader(pdf_url)
        docs = loader.load()
        
        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Create embeddings and store them in an in-memory Chroma database
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        
        return vectorstore.as_retriever(search_kwargs={"k": 5})

# Initialize retriever
retriever = initialize_vector_store()

# 4. Setup LLM and RAG Chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# Create a system prompt instructing the AI how to behave
system_prompt = (
    "You are an expert legal assistant specializing in India's Right to Information (RTI) Act.\n"
    "Answer the user's question using strictly the provided context below. "
    "If you do not know the answer or if it's not mentioned in the context, say exactly "
    "'I cannot find that information in the provided RTI handbook.' Do not make things up.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Combine everything into a working retrieval chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 5. Build the Chat History UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_query := st.chat_input("e.g., What is the time limit for disposing an RTI request?"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching handbook..."):
            response = rag_chain.invoke({"input": user_query})
            answer = response["answer"]
            st.markdown(answer)
            
    # Save assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
