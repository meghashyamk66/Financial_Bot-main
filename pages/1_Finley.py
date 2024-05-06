from http import client
import pinecone
import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from pinecone import Pinecone, ServerlessSpec
from typing import List
from langchain import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage
from langchain.docstore.document import Document
# from psx import tickers
# import datetime 

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from streamlit_chat import message


load_dotenv()
# Initialize Pinecone client
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    docs = [Document(page_content=text) for text in chunks]
    return docs


def get_vectorstore(docs):
    embeddings = OpenAIEmbeddings()

    
    vectorstore_from_docs = PineconeVectorStore.from_documents(
        docs,
        index_name="fibot",
        embedding=embeddings
    )

    
    return vectorstore_from_docs



def get_conversation_chain():
    
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore.from_existing_index(index_name="fibot", embedding=embeddings)
    
    retriever = vectorstore.as_retriever()

    # Initialize ChatAnthropic
    llm = ChatOpenAI()

    # Contextualize question
    contextualizeQSystemPrompt = """
    Given a chat history and the latest user question
    which might reference context in the chat history,
    formulate a standalone question which can be understood
    without the chat history. Do NOT answer the question, just
    reformulate it if needed and otherwise return it as is.
    """
    contextualizeQPrompt = ChatPromptTemplate.from_messages([
        ("system", contextualizeQSystemPrompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    historyAwareRetriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualizeQPrompt
    )

    # Answer question
    qaSystemPrompt = """
    You are a Financial assistant named Finley for question-answering tasks. Try to
    understand the user's risk appetite and his/her financial goals and answer accordingly. 
    Use the following pieces of retrieved context to answer the
    question. If you don't know the answer, just say that you
    don't know. 
    \n\n
    {context}
    """
    
    qaPrompt = ChatPromptTemplate.from_messages([
        ("system", qaSystemPrompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    questionAnswerChain = create_stuff_documents_chain(
        llm=llm,
        prompt=qaPrompt
    )

    ragChain = create_retrieval_chain(
        retriever=historyAwareRetriever,
        combine_docs_chain=questionAnswerChain
    )

    return ragChain





def submit():
    st.session_state.something = st.session_state.widget
    st.session_state.widget = ''


def handle_userinput(user_question):
   
    # response = st.session_state.conversation({'question': user_question})
    
    chat_history = st.session_state.get("chat_history", [])
    print(chat_history)
    response = st.session_state.conversation.invoke(
        {'input': user_question,
         'chat_history': chat_history}
    )
    print("YOOHOO")
    print(response)
    
    new_messages = [HumanMessage(content=user_question), response["answer"]]
    chat_history.extend(new_messages)
    
    st.session_state.chat_history = chat_history
    
    


    # for i, message in enumerate(st.session_state.chat_history):
    #     if i % 2 == 0:
    #         st.write(user_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)
    #     else:
    #         st.write(bot_template.replace(
    #             "{{MSG}}", message), unsafe_allow_html=True)
            



def main():
    load_dotenv()
    if "conversation" not in st.session_state:
        
        st.session_state.conversation = get_conversation_chain()
        
    chathistory: List[BaseMessage] = []
        
    if "chat_history" not in st.session_state:
        print("Insideeeeeeee")
        st.session_state.chat_history = chathistory

    st.title("Chat with Finley :dollar:")

    chat_placeholder = st.empty()

    # with chat_placeholder.container():    
    #     for i, my_message in enumerate(st.session_state.chat_history):
    #         print("my_message: ",my_message)
    #         if i % 2 == 0:
    #             # st.write(user_template.replace(
    #             #     "{{MSG}}", message.content), unsafe_allow_html=True)
    #             message(my_message.content, is_user=True, key=f"{i}_user")
    #         else:
    #             message(
    #                 my_message, 
    #                 key=f"{i}",
    #                 )
  

    with st.container():
        user_question = st.text_input("Ask a questions on financials:")
        if user_question:
            handle_userinput(user_question)


    with chat_placeholder.container():    
        for i, my_message in enumerate(st.session_state.chat_history):
            print("my_message: ",my_message)
            if i % 2 == 0:
                message(my_message.content, is_user=True, key=f"{i}_user")
            else:
                message(
                    my_message, 
                    key=f"{i}",
                    )

    # load_dotenv()
    # st.set_page_config(page_title="Chat with multiple PDFs",
    #                    page_icon=":books:")
    # st.write(css, unsafe_allow_html=True)

    # if "conversation" not in st.session_state:
        
    #     st.session_state.conversation = get_conversation_chain()
        
    # chathistory: List[BaseMessage] = []
        
    # if "chat_history" not in st.session_state:
    #     print("Insideeeeeeee")
    #     st.session_state.chat_history = chathistory


    # st.header("Chat with Finley :dollar:")
    # user_question = st.text_input("Ask a questions on financials:")
    # if user_question:
    #     handle_userinput(user_question)


    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your neccessary documents here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                vectorstore = get_vectorstore(text_chunks)
                

if __name__ == '__main__':
    main()
