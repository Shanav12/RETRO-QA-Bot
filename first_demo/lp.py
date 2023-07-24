import streamlit as st
from htmlTemplates import css, bot_template, user_template
import pinecone
import os
import logging
import PyPDF2
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
import shutil
import os
import tempfile
import pypdf
import openai
from llama_index import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
)

from llama_index import get_response_synthesizer
from llama_index.query_engine import RetrieverQueryEngine

# import QueryBundle
from llama_index import QueryBundle

# import NodeWithScore
from llama_index.schema import NodeWithScore

# Retrievers
from llama_index.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)

from typing import List
from llama_index.node_parser import SimpleNodeParser

from llama_index import GPTVectorStoreIndex, StorageContext, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores import PineconeVectorStore
from llama_index import Document

OPEN_AI_KEY =  # insert 
PINECONE_API_KEY = # insert 
PINECONE_API_ENV = # insert 


# Function to get the text from PDFs
def get_pdf_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to get the documents using SimpleDirectoryReader

def get_documents(pdf_names, pdfs):
    documents = []
    for i in range(len(pdfs)):
        # File exists, continue with your code logic
        temp_dir = tempfile.mkdtemp()  # Create a temporary directory
        temp_file_path = os.path.join(temp_dir, pdf_names[i])

        # Copy the file to the temporary directory
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(pdfs[i].getbuffer())

        # Use the temporary directory path with the SimpleDirectoryReader
        documents.extend(SimpleDirectoryReader(temp_dir).load_data())

        # Cleanup the temporary directory
        shutil.rmtree(temp_dir)

    return documents

# gets the nodes
def node(documents):
  parser = SimpleNodeParser()
  nodes = parser.get_nodes_from_documents(documents)
  return nodes

# Function to create the Pinecone index
def create_index(documents):
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

    # Setting the index name
    index_name = 'your_index_name'

    # Connect to the index
    pinecone_index = pinecone.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    # Setting up our vector store (Pinecone)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Setup the index process which we will use to query our documents
    embedding_model = OpenAIEmbedding(model='text-embedding-ada-002', embed_batch_size=100)

    service_context = ServiceContext.from_defaults(embed_model=embedding_model)

    index = GPTVectorStoreIndex.from_documents(
        documents, storage_context=storage_context, service_context=service_context
    )

    return index




# Function to create the conversation chain
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=OPEN_AI_KEY)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


# Function to handle user input and have a conversation
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


# Main function
def main():
    st.set_page_config(page_title="RETRO")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("RETRO")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                file_names = []
                if pdf_docs:
                    file_names = [file.name for file in pdf_docs]

                # Get pdf text
                #raw_text = get_pdf_text(file_names)

                documents = get_documents(file_names, pdf_docs)

                #nodes = node(documents)

                st.session_state.conversation  = create_index(documents)


if __name__ == '__main__':
    main()
