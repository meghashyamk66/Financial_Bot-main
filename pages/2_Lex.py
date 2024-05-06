import json
from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import os
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template

load_dotenv()

api_key= os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)

def get_dividend_rate(ticker):
    return str(yf.Ticker(ticker).info['dividendRate'])

def get_return_on_equity(ticker):
    return str(yf.Ticker(ticker).info['returnOnEquity'])

def get_peg_ratio(ticker):
    return str(yf.Ticker(ticker).info['pegRatio'])



def give_investment_advice(ticker = ['AAPL', 'META', 'AMZN', 'NVDA', 'GOOGL', 'PYPL', 'QCOM', 'SBUX', 'CSCO', 'JAGX', 'EXPE'], extra= ['NVDA', 'GOOGL','QCOM']):
  # print(ticker)
  divident_rates = []
  return_on_equity = []
  peg_ratios = []
  err = "Error"
  not_tick = []
  if ticker is not None:

    for t in ticker:
      
      if 'dividendRate' in (yf.Ticker(t).info.keys()):
        divident_rates.append(f"{t}: {yf.Ticker(t).info['dividendRate']}")
      if 'returnOnEquity' in (yf.Ticker(t).info.keys()):
        return_on_equity.append(f"{t}: {yf.Ticker(t).info['returnOnEquity']}")
      if 'pegRatio' in (yf.Ticker(t).info.keys()):
        peg_ratios.append(f"{t}: {yf.Ticker(t).info['pegRatio']}")

      for e in extra:
        if 'dividendRate' in (yf.Ticker(e).info.keys()):
          divident_rates.append(f"{e}: {yf.Ticker(e).info['dividendRate']}")
        if 'returnOnEquity' in (yf.Ticker(e).info.keys()):
          return_on_equity.append(f"{e}: {yf.Ticker(e).info['returnOnEquity']}")
        if 'pegRatio' in (yf.Ticker(e).info.keys()):
          peg_ratios.append(f"{e}: {yf.Ticker(e).info['pegRatio']}")
    
    sorted_divident_rates = sorted(divident_rates, key=lambda x: float(x.split(':')[1].strip() or 0), reverse=True)
    sorted_return_on_equity = sorted(return_on_equity, key=lambda x: float(x.split(':')[1].strip() or 0), reverse=True)
    sorted_peg_ratios = sorted(peg_ratios, key=lambda x: float(x.split(':')[1].strip() or 0), reverse=True)

    top_three_divident_rates = '\n'.join(sorted_divident_rates[:3])
    top_three_return_on_equity = '\n'.join(sorted_return_on_equity[:3])
    top_three_peg_ratios = '\n'.join(sorted_peg_ratios[-4:-1])

  else:
      
       return not_tick.append(f"{err}: Error fetching price")

  return f"Top 3 stocks based on Divident Rates: {top_three_divident_rates}\nTop 3 stocks based on  Return on Equity: {top_three_return_on_equity}\nTop 3 stocks based on lowest Peg Ratios: {top_three_peg_ratios}"
   

def get_stock_news(ticker = 'AAPL'):
  all_news = yf.Ticker(ticker).news[0:10]
  data_dict = {}
  prices = yf.Ticker(ticker).history(period='1y').iloc[-4:-1].to_json()

  data = yf.Ticker(ticker).history(period='1y')
  plt.figure(figsize=(10, 5))
  plt.plot(data.index, data.Close)
  plt.title(f'{ticker} Stock Price Over Last Year')
  plt.xlabel('Date')
  plt.ylabel('Stock Price ($)')
  plt.grid(True)
  plt.savefig('stock.png')
  plt.close()


  for idx , news in enumerate(all_news):
     data_dict[idx] = news
  obj =json.dumps(data_dict)
  # print(prices)
  return f"Price 3 day summary {prices} that are to be shown to the user\n new update {obj}"

     

def get_batch_stock_quotes(ticker = ['AAPL', 'META', 'AMZN', 'NVDA', 'GOOGL', 'PYPL', 'QCOM', 'SBUX', 'CSCO', 'JAGX', 'EXPE']): 
    prices = []
    err = "Error"
    if ticker is not None:
      for ticker in ticker:
          try:
              # Fetch the historical data for the last year
              #print(yf.Ticker(ticker).info['forwardPE','dividendRate','returnOnEquity','pegRatio','52WeekChange','50DayMovingAverage','200DayMovingAverage'])
              #print(yf.Ticker(ticker).info['overallRisk'])
              #print((yf.Ticker(ticker).news))
              # print(yf.Ticker(ticker).news)
              
              close_price = yf.Ticker(ticker).history(period='1y').iloc[-1].Close
              prices.append(f"{ticker}: {close_price}")
          except Exception as e:
              # If there's an error (e.g., no data available), record it
              prices.append(f"{ticker}: Error fetching price")
          
      # Join all ticker prices into a single string
      # sorted_prices = sorted(prices, key=lambda x: float(x.split(':')[1].strip() or 0), reverse=True)
      # print(sorted_prices)
      else:
          prices.append(f"{err}: Error fetching price")
          
    return '\n'.join(prices)


def get_stock_price(ticker, years=[2024], Months = []):
    data_dict = {}
    if len(years) == 1 and years[0] < 2000:
       obj = str(yf.Ticker(ticker).history(period='1y').iloc[-1].Close)

    for year in years:
        if year != 2024:
          data_dict[year] = yf.Ticker(ticker).history(start=f"{year}-01-01", end=f"{year}-03-31").iloc[::10].to_json()
        else:
            data_dict[year] = yf.Ticker(ticker).history(start=f"{year}-01-01", end=None).to_json()
    obj = json.dumps(data_dict)
    return obj
def calculate_SMA(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.rolling(window=window).mean().iloc[-1])

def calculate_EMA(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.ewm(span=window, adjust=False).mean().iloc[-1])

def calculate_RSI(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=14-1, adjust=False).mean()
    ema_down = down.ewm(com=14-1, adjust=False).mean()
    rs = ema_up / ema_down
    return str(100 - (100 / (1+rs)).iloc[-1])

def calculate_MACD(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    short_EMA = data.ewm(span=12, adjust=False).mean()
    long_EMA = data.ewm(span=26, adjust=False).mean()
    macd = short_EMA - long_EMA
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_histogram = macd-signal
    return f'{macd[-1]}, {signal[-1]}, {macd_histogram[-1]}'

def plot_stock_price(ticker):
    data = yf.Ticker(ticker).history(period='1y')
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data.Close)
    plt.title(f'{ticker} Stock Price Over Last Year')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.grid(True)
    plt.savefig('stock.png')
    plt.close()

def check_input(input_str):
    if "investment" in input_str.split(' ') or 'advise' in input_str.split(' '):
        return True
    
    else:
        return False
    
    

functions =  [
    {
      "name": "get_stock_price",
      "description": "Gets the latest stock prices given the ticker symbol of a company.",
      "parameters": {
        "type": "object",
        "properties": {
          "ticker": {
            "type": "string",
            "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)."
          },
          "years": {
            "type": "array",
            "items": {
                "type": "number"
            }
          }
        },
       "required": ["ticker", "years"],
      },
        "returns": {
            "type": "string",
            "description": "json formatted string with years and stock prices."
        }
    },
    {
      "name": "calculate_SMA",
      "description": "Calculate the simple moving average for a given stock ticker and a window.",
      "parameters": {
        "type": "object",
        "properties": {
          "ticker": {
            "type": "string",
            "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)."
          },
          "window": {
            "type": "integer",
            "description": "The timeframe to consider when calculating the SMA."
          }
        },
        "required": ["ticker", "window"]
      }
    },
    {
      "name": "calculate_EMA",
      "description": "Calculate the exponential moving average for a given stock ticker and a window.",
      "parameters": {
        "type": "object",
        "properties": {
          "ticker": {
            "type": "string",
            "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)."
          },
          "window": {
            "type": "integer",
            "description": "The timeframe to consider when calculating the EMA."
          }
        },
        "required": ["ticker", "window"]
      }
    },
    {
      "name": "calculate_RSI",
      "description": "Calculate the RSI for a given stock ticker.",
      "parameters": {
        "type": "object",
        "properties": {
          "ticker": {
            "type": "string",
            "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)."
          },
        },
        "required": ["ticker"]
      }
    },
    {
      "name": "calculate_MACD",
      "description": "Calculate the MACD for a given stock ticker",
      "parameters": {
        "type": "object",
        "properties": {
          "ticker": {
            "type": "string",
            "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)."
          },
        },
        "required": ["ticker"]
      }
    },
    {
      "name": "plot_stock_price",
      "description": "Plot the stock price for the last year given the ticker symbol of a company.",
      "parameters": {
        "type": "object",
        "properties": {
          "ticker": {
            "type": "string",
            "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)."
          }
        },
        "required": ["ticker"]
      }
    },
    {
      "name": "get_batch_stock_quotes",
      "description": "Retrieves the latest stock prices for a list of ticker symbols using yfinance and returns them as a formatted string. Note: This function only works is user asks for prices",
      "parameters": {
        "type": "object",
        "properties": {
          "ticker": {
            "type": "array",
            "items": {
              "type": "string"
            },
            "description": "A list of stock ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])."
          }
        }
      },
      "returns": {
        "type": "string",
        "description": "A string with each ticker symbol and its corresponding price, formatted for easy reading (e.g., 'AAPL: $150, MSFT: $250')."
      }
    },
    {
      "name": "give_investment_advice",
      "description": "Gives investment advise by retrieving information of the tickers and gives the top three companies that have the highest dividend rate, return on equity, and lowest peg ratio. Note: this function only works if user asks for advise regarding investing in a stock.",
      "parameters": {
        "type": "object",
        "properties": {
          "ticker": {
            "type": "array",
            "items": {
              "type": "string"
            },
            "description": "A list of stock ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])."
          }
        }
      },
      "returns": {
        "type": "string",
        "description": "A string that gives the top three companies that have the highest dividend rate, return on equity, and lowest peg ratio respectively, and then gives investment advise. (eg. 'The top three companies that have the highest dividend rate are meta platform, apple, and Marvell Technology Inc., top three companies that have the highest return on equity are google Amazon.com Inc, and Marvell Technology Inc., top three companies that have the lowest peg ratio are Apple Inc, Amazon.com Inc, and Marvell Technology Inc.')"
      }
    },

    {
      "name": "get_stock_news",
      "description": "Gets the latest stock prices and news given the ticker symbol of a company. Note: use if user asks for information, news or summary of a particular stock.",
      "parameters": {
        "type": "object",
        "properties": {
          "ticker": {
            "type": "string",
            "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)."
          }
        },
       "required": ["ticker"],
      },
        "returns": {
            "type": "string",
            "description": "String with json formatted price history of 3 days, and json formatted new update. Note: always provide the user with ths price summary, and articles"
        }
    }
   
  ]

available_functions= {
    'get_stock_price': get_stock_price,
    'calculate_SMA': calculate_SMA,
    'calculate_EMA': calculate_EMA,
    'calculate_RSI': calculate_RSI,
    'calculate_MACD': calculate_MACD,
    'plot_stock_price': plot_stock_price,
    'get_batch_stock_quotes': get_batch_stock_quotes,
    'give_investment_advice': give_investment_advice,
    'get_stock_news': get_stock_news
}




if 'messages' not in st.session_state:
    st.session_state['messages']= []

if 'chat_history' not in st.session_state:
  st.session_state['chat_history']= []

st.title("Stock analysis Chatbot")
st.write(css,unsafe_allow_html=True)

user_input = st.text_input('Your input: ')
# print(user_input)

if user_input:
    try:
        
        check_ip = check_input(user_input)

        # if check_input:
        
        #   st.session_state['messages'].append({'role':'user', 'content': f"{user_input} if the user has asked about a pecific company, use that company's information, else also use the following stocks in your calculations to make them more hollistic: Apple Inc, Meta Platforms Inc, Amazon.com Inc, NVIDIA Corporation, Alphabet Inc, PayPal Holdings Inc,  Qualcomm Inc,Starbucks Corporation, Cisco Systems Inc, Jaguar Health Inc, and Expedia Group Inc. "})
        # else:
        st.session_state['chat_history'].append({'role': 'user', 'content': user_input})
        st.session_state['messages'].append({'role':'user', 'content': f'{user_input} \n As an AI chatbot, answer the question asked above.'})

        response = client.chat.completions.create(
            model= 'gpt-3.5-turbo',
            messages= st.session_state['messages'],
            functions=functions,
            function_call='auto',
        )
        
        response_message =  response.choices[0].message
        # print("response message: ", response_message)
        # print( response_message.function_call)
        if response_message.function_call:
            function_name = response_message.function_call.name
            function_args = json.loads(response_message.function_call.arguments)

            if function_name in ['get_stock_price', 'calculate_RSI', 'calculate_MACD', 'plot_stock_price', 'get_batch_stock_quotes','give_investment_advice','get_stock_news']:
                args_dict = {'ticker': function_args.get('ticker')}
            elif function_name in ['calculate_SMA', 'calculate_EMA']:
                args_dict = {'ticker': function_args.get('ticker'), 'window': function_args.get('window')}
            elif function_name in ['get_stock_price']:
                args_dict = {'ticker': function_args.get('ticker'), 'years': function_args.get('years')}

            function_to_call = available_functions[function_name]
            # print(function_to_call)
            function_response = function_to_call(**args_dict)

            if function_name == 'plot_stock_price':
                st.image('stock.png')
            else:
                st.session_state['messages'].append(response_message)
                st.session_state['messages'].append({
                    'role': 'function',
                    'name': function_name,
                    'content': function_response
                })

            second_response= client.chat.completions.create(
                model= 'gpt-3.5-turbo',
                messages= st.session_state['messages'],
            )

            my_chat = st.text_area(label= "Answer: " , value = second_response.choices[0].message.content, height =100)
            st.session_state['chat_history'].append({'role': 'bot', 'content':my_chat})
            st.session_state['messages'].append({'role': "system", "content": "You are an expert stock market analyst.when asked for stock news, summary, or infrmation, give the price history. You have the following nasdaq companies American Airlines Group Inc, Apple Inc, AGNC Investment Corporation, Akanda Corporation, Advanced Micro Devices Inc, Amgen Inc, Amazon.com Inc, APA Corporation, Ardelyx Inc, Aurora Innovations Inc, AXT Inc, AstraZeneca PLC, Beneficient, Bitfarms Ltd, Bruush Oral Care Inc, Canaan Inc, Canopy Growth Corporation, CleanSpark Inc, Comcast Corporation, Coinbase Global Inc, Crown Electrokinetics Corporation, Cisco Systems Inc, CytomX Therapeutics Inc, CYNGN Inc, DraftKings Inc, Enovix Corporation, Ericsson, Expedia Group Inc, FuelCell Energy Inc, Faraday Future Intelligent Electric Inc, Fortinet Inc, Alphabet Inc, Alphabet Inc, Grab Holdings Ltd, Greenwave Technology Solutions Inc, Huntington Bancshares Inc, Healthcare Triangle Inc, MicroCloud Hologram Inc, Robinhood Markets Inc, Helius Medical Technologies Inc, Hertz Global Holdings Inc, IAC Inc, iShares Bitcoin Trust ETF, ImmunityBio Inc, Intel Corporation, iQiyi Inc, Inspire Veterinary Partners Inc, Jaguar Health Inc, JetBlue Airways Corporation, JD com Inc, Jeffs Brands Ltd, Luminar Technologies Inc, Lucid Group Inc, Li Auto Inc, Lyft Inc, Marathon Digital Holdings Inc, Meta Platforms Inc, Mobile health Network Solutions, Monster Beverage Corporation, Marvell Technology Inc, Microsoft Corporation, Micron Technology Inc, Nikola Corporation, Newellis Inc, NVIDIA Corporation, Pacific Biosciences of California Inc, Paramount Global, Plug Power Inc, Peloton Interactive Inc, PayPal Holdings Inc, QUALCOMM Inc, Sunrun Inc, Starbucks Corporation, Tesla Inc \n" 'assistant', 'content': second_response.choices[0].message.content})
        else:
            the_chat = st.text_area(label= "Answer: " , value = response_message.content, height =100)
            st.session_state['chat_history'].append({'role': 'bot', 'content':the_chat})
            st.session_state['messages'].append({'role': 'assistant', 'content': response_message.content})
    except Exception as e:
        raise(e)
    
for i, message in enumerate(st.session_state.chat_history): 
  if i % 2 == 0:
      st.write(user_template.replace("{{MSG}}",message['content']), unsafe_allow_html=True)
  else:
      st.write(bot_template.replace(
          "{{MSG}}", message['content']), unsafe_allow_html=True)