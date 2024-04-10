import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Google API key
os.environ['GOOGLE_API_KEY'] =  "YOUR_GOOGLE_API_KEY_HERE"

# Function to extract text from PDF
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

# Function to create embeddings and vector store
def get_embed(chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # print("Length of embeddings:", len(embeddings))
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        print("Error creating FAISS index:", e)
        raise

def get_conversational_chain():
    prompt_template = """
     You are a resume builder assistant. You need to analyze the input given by the user and evaluate his resume. You should also give him advice on building up a resume and making necessary changes. Help the user in every possible manner. Give a detailed review and scope of improvements if the answer is not in provided context, just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        response = user_input(user_question)
        st.write("Reply: ", response)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_embed(text_chunks)
                st.success("Done")
pdf_path="C:/Users/susha/OneDrive/Desktop/engineering/python(theory)/python vs/sushant-resume.pdf"
text=get_pdf_text(pdf_path)
chunks=get_text_chunks(text)
get_embed(chunks)
user_input("please analyze my resume")
# if __name__ == "__main__":
#     main()
