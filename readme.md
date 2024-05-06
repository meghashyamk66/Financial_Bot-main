# Stock Analysis Chatbot (Finley & Lex)

## Description

This project implements a stock analysis chatbot that provides various functionalities related to stock analysis and financial advice. It leverages multiple APIs and libraries including OpenAI, yfinance, Streamlit, and Pinecone for natural language processing, stock data retrieval, and conversation management.

## Features

Stock Price Retrieval: Get the latest stock price given the ticker symbol of a company.
Simple Moving Average (SMA) Calculation: Calculate the simple moving average for a given stock ticker and window.
Exponential Moving Average (EMA) Calculation: Calculate the exponential moving average for a given stock ticker and window.
Relative Strength Index (RSI) Calculation: Calculate the RSI for a given stock ticker.
Moving Average Convergence Divergence (MACD) Calculation: Calculate the MACD for a given stock ticker.
Stock Price Plotting: Plot the stock price for the last year given the ticker symbol of a company.
Batch Stock Quotes Retrieval: Retrieves the latest stock prices for a list of ticker symbols and returns them as a formatted string.
Investment Advice Generation: Gives investment advice by analyzing dividend rates, return on equity, and peg ratios of specified stocks.

## Setup Instructions

Clone the repository: `git clone <repository_url>`
Install dependencies: `pip install -r requirements.txt`
Set up environment variables: Create a .env file and add your API keys for OpenAI and Pinecone.
makefile
Copy code
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
Run the application: `streamlit run hellp.py`

## Usage

Enter your input in the provided text input field.
The chatbot will process your input and provide relevant responses based on the implemented functionalities.
You can interact with different features of the chatbot such as retrieving stock prices, calculating technical indicators, plotting stock prices, and receiving investment advice.

## Contributors

Azmeer Faisal
Nameer Anjum
Muhammad Umer Malik
Muhammed Salman Asif
Idrees Aziz
Muhammed Talal Siddiqui
