from langchain import OpenAI 
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import Tool
from langchain.tools import BaseTool
import json
from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
# from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
import streamlit as st
from langchain import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain


turbo_llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo'
)

def get_stock_price(ticker):
    return str(yf.Ticker(ticker).history(period='1y').iloc[-1].Close)

stock_price = Tool(
    name='Stock Price',
    func= get_stock_price,
    description="Gets realtime stock price, The stock ticker symbol for a company (e.g., AAPL for Apple)."
)

embeddings = OpenAIEmbeddings()
vectordb = PineconeVectorStore.from_existing_index(index_name="fibot", embedding=embeddings)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)



# Create Prompt
template = """You are a Financial assistant named Finley for question-answering tasks. Try to
    understand the user's risk appetite and his/her financial goals and answer accordingly. 
    Use the following pieces of retrieved context to answer the
    question. If you don't know the answer, just say that you
    don't know.
{context}


Question: {question}
Answer: 
"""


prompt = PromptTemplate.from_template(template)







retriever = RetrievalQA.from_chain_type(
    llm=turbo_llm,
    chain_type="stuff",
    retriever= vectordb.as_retriever(),
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt},
    verbose= True
)



########################################################
def get_conversation_chain():
  
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore.from_existing_index(index_name="fibot", embedding=embeddings)
    
    retriever = vectorstore.as_retriever()


    llm = ChatOpenAI()

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
    understand the user's risk appetite and his/her financial goals and answer accordingly. Try to give descriptive answers analysing all possible situations.
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





########################################################































tools = [stock_price,Tool(
        func=retriever.run, description="Anything else expect for stock price", name="Custom Data"
    ),]


# conversational agent memory
memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=15,
    return_messages=True
)

print("hererererere")


# create our agent
conversational_agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=turbo_llm,
    verbose=True,
    # max_iterations=3,
    # early_stopping_method='generate',
    memory=memory
)

sys_msg = """You are a Financial assistant named Finley for question-answering tasks. If the user asks about investment opportunities understand the user's risk appetite and his/her financial goals by answering a series of relevant questions using the context provided and give appropriate answers using the context. If the user wants help in calculating his income tax ask him a series of relevant questions and help him calulate his total income tax (assuming he lives in Pakistan) using the context. Try to get the answers to all relevant questions before calculating his income tax.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    """

prompt = conversational_agent.agent.create_prompt(system_message=sys_msg, tools=tools)
conversational_agent.agent.llm_chain.prompt = prompt

def main():
    st.title("Stock Analysis Chatbot")

    # Initialize the conversational agent only once using session state
    if 'conversational_agent' not in st.session_state:
        st.session_state.conversational_agent = initialize_agent(
            agent='chat-conversational-react-description',
            tools=tools,
            llm=turbo_llm,
            verbose=True,
            memory=memory
        )

    user_input = st.text_input('Your input: ')

    if user_input:
        try:
            # Use the conversational agent from session state
            answer = st.session_state.conversational_agent(user_input)
            # st.text(answer['output'])
            st.text_area(label="Output Data:", value=answer['output'], height=350)
            print("answer: ", answer)
        except Exception as e:
            st.error("An error occurred: {}".format(e))

if __name__ == '__main__':
    main()



# answer = conversational_agent("Get me stock price for Apple")
# print("answer: ", answer)

# answer2 = conversational_agent("Get me stock price for Samsung")
# print("answer2: ", answer2)

# answer3 = conversational_agent("Tell me about azmeer faisal")
# print("answer2: ", answer3)

# answer4 = conversational_agent("Tell me more about him")
# print("answer2: ", answer4)

