from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
import os
import requests

load_dotenv()

app = Flask(__name__)
socketio = SocketIO(app)  
@app.route('/qa', methods=['POST'])
def question_answer():
    """Perform a question-answer task using a PDF document and a query.

    This function takes a PDF URL and a query as input. It downloads the
    PDF, extracts text from the PDF pages, generates embeddings for the
    text, and then searches for relevant documents based on the query. It
    then uses a question-answering model to find the answer to the query
    from the relevant documents.

    Returns:
        dict: A dictionary containing the response to the query.
    """

    try:
      
        request_data = request.get_json()
        pdf_url = request_data.get('pdf_url', '')
        query = request_data.get('query', '')

        pdf_filename = 'downloaded_pdf.pdf'  
        response = requests.get(pdf_url)
        with open(pdf_filename, 'wb') as pdf_file:
            pdf_file.write(response.content)
        loader = UnstructuredPDFLoader(pdf_filename)
        pages = loader.load_and_split()

       
        embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_documents(pages, embeddings).as_retriever()

   
        #query = "What is the amount of Total Monthly Service Fees?"
        docs = docsearch.get_relevant_documents(query)
      
        chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff")
        output = chain.run(input_documents=docs, question=query)
        return jsonify({'response': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    socketio.run(app)  