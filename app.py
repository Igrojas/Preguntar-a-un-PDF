import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def get_pdf(doc_pdf):
    text = ""
    pdf_read = PdfReader(doc_pdf)
    for page in pdf_read.pages:
        text += page.extract_text()
    return text

def get_chunks(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function=len
    )
    chunks = text_splitter.split_text(pdf_text)
    return chunks


def get_vectors(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store= FAISS.from_texts(text_chunks, embedding= embeddings)
    vector_store.save_local("faiss_index")


def chain_conversation():

    prompt_template = """Antes de contestar a las preguntas que te hagan, debes primero hacer lo siguiente:
                        Un resumen muy detallado del documento en no más de 3 parrafos, luego debes dar un bullets de las ideas claves, y finalmente un bullet de prguntas claves de estudío
                        
                    Context: \n {context}? \n
                    Question: \n {question}\n

                    Answer:
                        """


    model = ChatGoogleGenerativeAI(model= 'gemini-pro',
                               temperature= 0.5)
    prompt = PromptTemplate(template= prompt_template)
    chain= load_qa_chain(model, chain_type='stuff', prompt = prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = chain_conversation()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat PDF")
    st.header("Haz una consulta a un PDF usando Gemini :D")

    user_question = st.text_input("Realiza una pregunta")

    if user_question:
        user_input(user_question)

    st.title("Menu:")
    pdf_docs = st.file_uploader("Sube tu archivo PDF")
    if st.button("Enviar"):
        with st.spinner("Espere..."):
            raw_text = get_pdf(pdf_docs)
            text_chunks = get_chunks(raw_text)
            get_vectors(text_chunks)
            st.success("Listo :D")



if __name__ == "__main__":
    main()

