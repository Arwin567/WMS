import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFium2Loader
from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI


class PDFQuery:
    def __init__(self, openai_api_key = None) -> None:
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
        self.chain = None
        self.db = None

    def ask(self, question: str) -> str:
        """Responds to a question based on relevant documents in the database.

        If a document chain is not set, it prompts to add a document. Otherwise,
        it retrieves relevant documents from the database and processes the
        question using the document chain.

        Args:
            question (str): The question to respond to.

        Returns:
            str: The response to the input question.
        """

        if self.chain is None:
            response = "Please, add a document."
        else:
            docs = self.db.get_relevant_documents(question)
            response = self.chain.run(input_documents=docs, question=question)
        return response

    def ingest(self, file_path: os.PathLike) -> None:
        """Ingests a PDF file, processes the text, and creates a retriever for
        further operations.

        Args:
            file_path (os.PathLike): Path to the PDF file to be ingested.
        """

        loader = PyPDFium2Loader(file_path)
        documents = loader.load()
        splitted_documents = self.text_splitter.split_documents(documents)
        self.db = Chroma.from_documents(splitted_documents, self.embeddings).as_retriever()
        # self.chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
        self.chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff")

    def forget(self) -> None:
        """Reset the database and blockchain references to None.

        This method resets the database and blockchain references to None,
        effectively forgetting any existing data.
        """

        self.db = None
        self.chain = None