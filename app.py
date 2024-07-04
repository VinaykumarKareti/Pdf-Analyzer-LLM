
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Set environment variable to avoid OpenMP runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
def get_conversational_chain(question,context):
  prompt_template = f'''
    Assume that you are a intelligent in answering the question based on the context given in the below.
    Task: generate the answer in detail based on the below question and context.
          Note: just give the generated answer as response and not include the question, context and other in the response.
    Question: \n${question}\n
    Context:\n ${context}?\n
    
    Answer:

    '''
  inputs = tokenizer(prompt_template, return_tensors="pt")
  outputs = model.generate(inputs['input_ids'], max_new_tokens=500, num_return_sequences=1)
  generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
  answer_part = generated_text.split("Answer:")[-1].strip()
  answer_part.replace("\n","")
  return answer_part

# Function to process user input
def user_input(user_question, conversation_history):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    res = get_conversational_chain(user_question, docs)
    conversation_history.append({"question": user_question, "answer": res})

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Chat with PDF By Vinay Kumar💁")
    st.header("Chat with PDF by Vinay Kumar💁")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Display conversation history
    st.subheader("Conversation:")
    for chat in st.session_state.conversation_history:
        st.write(f"**You:** {chat['question']}")
        st.write(f"**Vinay:** {chat['answer']}")

    # Input for user question
    user_question = st.text_input("Ask a Question from the PDF Files")

    # Button to submit user question
    if st.button("Send"):
        user_input(user_question, st.session_state.conversation_history)
        st.experimental_rerun()

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")
    main()

