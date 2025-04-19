import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile
import pickle
from azure.storage.blob import BlobServiceClient
from io import BytesIO

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()
 
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
 
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONN_STRING")
CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")
VECTOR_STORE_PATH = "vector_store/faiss_index"
PROCESSED_FILES_LIST_PATH = "vector_store/processed_files.pkl"
 
def get_blob_client():
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        return container_client
    except Exception as e:
        st.error(f"Error connecting to Azure Blob Storage: {str(e)}")
        return None
 
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf) 
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
 
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks
 
def blob_exists(container_client, blob_path):
    try:
        blob_client = container_client.get_blob_client(blob_path)
        return blob_client.exists()
    except Exception:
        return False
 
def save_to_blob(container_client, blob_path, data):
    try:
        blob_client = container_client.get_blob_client(blob_path)
         
        blob_client.upload_blob(data, overwrite=True)
        return True
    except Exception as e:
        st.error(f"Error saving to Azure Blob Storage: {str(e)}")
        return False
 
def download_from_blob(container_client, blob_path):
    try:
        blob_client = container_client.get_blob_client(blob_path)
        if not blob_client.exists():
            return None
        
        download = blob_client.download_blob()
        downloaded_bytes = download.readall()
        return downloaded_bytes
    except Exception as e:
        st.error(f"Error downloading from Azure Blob Storage: {str(e)}")
        return None
 
def get_processed_files(container_client):
    processed_files = []
    data = download_from_blob(container_client, PROCESSED_FILES_LIST_PATH)
    if data:
        processed_files = pickle.loads(data)
    return processed_files
 
def save_processed_files(container_client, processed_files):
    data = pickle.dumps(processed_files)
    return save_to_blob(container_client, PROCESSED_FILES_LIST_PATH, data)
 
def get_vector_store(text_chunks, container_client, append=False):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    
    if append and blob_exists(container_client, VECTOR_STORE_PATH): 
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_index_path = os.path.join(temp_dir, "faiss_index")
             
            index_files_data = download_from_blob(container_client, VECTOR_STORE_PATH)
            if index_files_data:
                data_dict = pickle.loads(index_files_data)
                 
                with open(os.path.join(temp_dir, "index.faiss"), "wb") as f:
                    f.write(data_dict["index.faiss"])
                with open(os.path.join(temp_dir, "index.pkl"), "wb") as f:
                    f.write(data_dict["index.pkl"])
                 
                vector_store = FAISS.load_local(temp_dir, embeddings)
                 
                vector_store.add_texts(text_chunks)
            else:
                vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    else: 
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
     
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store.save_local(temp_dir)
         
        with open(os.path.join(temp_dir, "index.faiss"), "rb") as f:
            index_data = f.read()
        with open(os.path.join(temp_dir, "index.pkl"), "rb") as f:
            pkl_data = f.read()
         
        combined_data = pickle.dumps({"index.faiss": index_data, "index.pkl": pkl_data})
        return save_to_blob(container_client, VECTOR_STORE_PATH, combined_data)
 
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
 
    #model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

    model = ChatGroq(model="llama3-70b-8192", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain
 
def local_user_input(user_question, vector_store): 
    docs = vector_store.similarity_search(user_question)
    
    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    return response["output_text"]
 
def blob_user_input(user_question, container_client):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
     
    with tempfile.TemporaryDirectory() as temp_dir: 
        combined_data = download_from_blob(container_client, VECTOR_STORE_PATH)
        if not combined_data:
            return "No documents have been stored in Azure Storage yet. Please upload and save documents first."
        
        data_dict = pickle.loads(combined_data)
         
        with open(os.path.join(temp_dir, "index.faiss"), "wb") as f:
            f.write(data_dict["index.faiss"])
        with open(os.path.join(temp_dir, "index.pkl"), "wb") as f:
            f.write(data_dict["index.pkl"])
         
        vector_store = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
         
        docs = vector_store.similarity_search(user_question)

        chain = get_conversational_chain()
        
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        
        return response["output_text"]
 
def upload_and_query_pdf():
    st.header("Upload & Query PDF")
     
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
     
    pdf_docs = st.file_uploader("Upload your PDF Files and Click Process", accept_multiple_files=True)
    
    if st.button("Process"):
        if pdf_docs:
            with st.spinner("Processing..."): 
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                
                embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
                st.session_state.vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
                
                st.success("PDFs processed successfully! You can now ask questions.")
                 
                st.session_state.current_pdfs = [pdf.name for pdf in pdf_docs] 
                st.session_state.pdf_contents = {pdf.name: pdf.getvalue() for pdf in pdf_docs}
        else:
            st.warning("Please upload PDF files to process.")
     
    if st.session_state.vector_store is not None:
        user_question = st.text_input("Ask a question about your PDFs:")
        if user_question:
            with st.spinner("Searching..."):
                response = local_user_input(user_question, st.session_state.vector_store)
                st.write("Response:")
                st.write(response)
         
        st.subheader("Save to Azure Storage")
        save_to_azure_option = st.checkbox("Save documents and embeddings to Azure")
        
        if save_to_azure_option and st.button("Save to Azure"):
            container_client = get_blob_client()
            if container_client:
                with st.spinner("Saving to Azure Storage..."): 
                    pdf_save_success = True
                    vector_save_success = False
                    metadata_save_success = False
                     
                    for pdf_name, pdf_content in st.session_state.pdf_contents.items():
                        pdf_path = f"pdfs/{pdf_name}"
                        if not save_to_blob(container_client, pdf_path, pdf_content):
                            pdf_save_success = False
                            break
                    
                    if pdf_save_success: 
                        processed_files = get_processed_files(container_client)
                         
                        new_files_added = False
                        for pdf_name in st.session_state.current_pdfs:
                            if pdf_name not in processed_files:
                                processed_files.append(pdf_name)
                                new_files_added = True
                         
                        if new_files_added:
                            metadata_save_success = save_processed_files(container_client, processed_files)
                        else:
                            metadata_save_success = True
                         
                        with tempfile.TemporaryDirectory() as temp_dir:
                            st.session_state.vector_store.save_local(temp_dir)
                             
                            with open(os.path.join(temp_dir, "index.faiss"), "rb") as f:
                                index_data = f.read()
                            with open(os.path.join(temp_dir, "index.pkl"), "rb") as f:
                                pkl_data = f.read()
                             
                            if blob_exists(container_client, VECTOR_STORE_PATH): 
                                existing_data = download_from_blob(container_client, VECTOR_STORE_PATH)
                                if existing_data: 
                                    with tempfile.TemporaryDirectory() as temp_dir2:
                                        data_dict = pickle.loads(existing_data)
                                         
                                        with open(os.path.join(temp_dir2, "index.faiss"), "wb") as f:
                                            f.write(data_dict["index.faiss"])
                                        with open(os.path.join(temp_dir2, "index.pkl"), "wb") as f:
                                            f.write(data_dict["index.pkl"])
                                         
                                        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
                                        existing_store = FAISS.load_local(temp_dir2, embeddings, allow_dangerous_deserialization=True)
                                         
                                        existing_store.merge_from(st.session_state.vector_store)
                                         
                                        existing_store.save_local(temp_dir2)
                                         
                                        with open(os.path.join(temp_dir2, "index.faiss"), "rb") as f:
                                            index_data = f.read()
                                        with open(os.path.join(temp_dir2, "index.pkl"), "rb") as f:
                                            pkl_data = f.read()
                             
                            combined_data = pickle.dumps({"index.faiss": index_data, "index.pkl": pkl_data})
                            vector_save_success = save_to_blob(container_client, VECTOR_STORE_PATH, combined_data)
                     
                    if pdf_save_success and vector_save_success and metadata_save_success:
                        st.success("Documents and embeddings saved to Azure Storage successfully!")
                    else:
                        st.error("Failed to save all data to Azure Storage. Please check the error messages above.")
            else:
                st.error("Failed to connect to Azure Storage. Please check your connection string and container name.")
 
def chat_with_azure_data():
    st.header("Chat with Azure Storage Data")
     
    container_client = get_blob_client()
    if not container_client:
        st.error("Failed to connect to Azure Storage. Please check your connection string and container name.")
        return
     
    if not blob_exists(container_client, VECTOR_STORE_PATH):
        st.warning("No documents have been stored in Azure Storage yet. Please use the 'Upload & Query PDF' tab first to save documents.")
        return
     
    processed_files = get_processed_files(container_client)
    if processed_files:
        st.subheader("Documents available in Azure Storage:")
        for file_name in processed_files:
            st.write(f"- {file_name}")
     
    if "azure_messages" not in st.session_state:
        st.session_state.azure_messages = [{"role": "assistant", "content": "You can ask questions about documents stored in Azure Storage."}]
     
    for message in st.session_state.azure_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
     
    user_question = st.chat_input("Ask a question about Azure Storage documents:")
    if user_question: 
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.azure_messages.append({"role": "user", "content": user_question})
         
        with st.spinner("Searching Azure Storage data..."):
            response = blob_user_input(user_question, container_client)
             
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.azure_messages.append({"role": "assistant", "content": response})
     
    if st.button("Clear Chat History"):
        st.session_state.azure_messages = [{"role": "assistant", "content": "You can ask questions about documents stored in Azure Storage."}]
        st.rerun()
 
def main():
    st.set_page_config(page_title="PDF Chat with Azure Storage Integration", layout="wide")
    st.title("PDF Chat with Azure Storage Integration")
     
    tab1, tab2 = st.tabs(["Upload & Query PDF", "Chat with Azure Storage Data"])
    
    with tab1:
        upload_and_query_pdf()
    
    with tab2:
        chat_with_azure_data()

if __name__ == "__main__":
    main()
