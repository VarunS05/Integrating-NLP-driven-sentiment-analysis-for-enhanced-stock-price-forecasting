import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from PIL import Image
import pandas as pd
from textblob import TextBlob

news_df = pd.read_csv('model_in_backend.csv')

st.title('Stock Price Predictions')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')

def main():
    option = st.sidebar.selectbox('Make a choice', ['Visualize','Recent Data', 'Predict', 'Predict with news'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    elif option == 'Predict with news':
        news()
    else:
        predict()



@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df



option = st.sidebar.text_input('Enter a Stock Symbol', value='SPY')
option = option.upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration', value=3000)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)
if st.sidebar.button('Send'):
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %(start_date, end_date))
        download_data(option, start_date, end_date)
    else:
        st.sidebar.error('Error: End date must fall after start date')




data = download_data(option, start_date, end_date)
scaler = StandardScaler()

def tech_indicators():
    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    # Bollinger bands
    bb_indicator = BollingerBands(data.Close)
    bb = data
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    # Creating a new dataframe
    bb = bb[['Close', 'bb_h', 'bb_l']]
    # MACD
    macd = MACD(data.Close).macd()
    # RSI
    rsi = RSIIndicator(data.Close).rsi()
    # SMA
    sma = SMAIndicator(data.Close, window=14).sma_indicator()
    # EMA
    ema = EMAIndicator(data.Close).ema_indicator()

    if option == 'Close':
        st.write('Close Price')
        st.line_chart(data.Close)
    elif option == 'BB':
        st.write('BollingerBands')
        st.line_chart(bb)
    elif option == 'MACD':
        st.write('Moving Average Convergence Divergence')
        st.line_chart(macd)
    elif option == 'RSI':
        st.write('Relative Strength Indicator')
        st.line_chart(rsi)
    elif option == 'SMA':
        st.write('Simple Moving Average')
        st.line_chart(sma)
    else:
        st.write('Expoenetial Moving Average')
        st.line_chart(ema)


def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(10))



def predict():
    model = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'])
    num = st.number_input('How many days forecast?', value=5)
    num = int(num)
    if st.button('Predict'):
        if model == 'LinearRegression':
            engine = LinearRegression()
            model_engine(engine, num)
        elif model == 'RandomForestRegressor':
            engine = RandomForestRegressor()
            model_engine(engine, num)
        elif model == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
            model_engine(engine, num)
        elif model == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
            model_engine(engine, num)
        else:
            engine = XGBRegressor()
            model_engine(engine, num)


def model_engine(model, num):
    # getting only the closing price
    df = data[['Close']]
    # shifting the closing price based on number of days forecast
    df['preds'] = data.Close.shift(-num)
    # scaling the data
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # storing the last num_days data
    x_forecast = x[-num:]
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]

    #spliting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    # training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'r2_score: {r2_score(y_test, preds)} \
            \nMAE: {mean_absolute_error(y_test, preds)}')
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1

def news():
    st.title("")
    model = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'])
    num = st.number_input('How many days forecast?', value=5)
    news_about_ADM = st.text_area("What is the news about ADM today?")
    num = int(num)
    if st.button('Predict'):
        if model == 'LinearRegression':
            engine = LinearRegression()
            newsmodel_engine(engine, num, news_about_ADM)
        elif model == 'RandomForestRegressor':
            engine = RandomForestRegressor()
            newsmodel_engine(engine, num, news_about_ADM)
        elif model == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
            newsmodel_engine(engine, num, news_about_ADM)
        elif model == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
            newsmodel_engine(engine, num, news_about_ADM)
        else:
            engine = XGBRegressor()
            newsmodel_engine(engine, num, news_about_ADM)

# Function to get sentiment scores
def getSentimentScores(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    
    # Calculate negative, neutral, and positive scores
    negative_score = 1 - sentiment.polarity
    neutral_score = 1 - (sentiment.polarity + sentiment.polarity)
    positive_score = sentiment.polarity

    # Calculate compound score (similar to VADER)
    compound_score = sentiment.polarity * sentiment.subjectivity

    return {
        "Polarity": sentiment.polarity,
        "Subjectivity": sentiment.subjectivity,
        "Negative": negative_score,
        "Neutral": neutral_score,
        "Positive": positive_score,
        "Compound": compound_score
    }


def newsmodel_engine(model,num, news_about_ADM):
    data = news_df
    # getting only the closing price
    df = news_df
    # shifting the closing price based on number of days forecast
    df['preds'] = data.Close.shift(-num)
    # scaling the data
    x = df.drop(['Date','preds'], axis=1).values
    # storing the last num_days data
    x_forecast = x[-num:]
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    # training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'r2_score: {r2_score(y_test, preds)} \
            \nMAE: {mean_absolute_error(y_test, preds)}')
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1

    '''# training the model
    model.fit(x, y)

    stock_symbol = "AZN.L"  # Example stock symbol (Apple Inc.)
    stock_data = yf.download(stock_symbol, start="2024-02-01" ,end="2024-03-20")
    news_headline = news_about_ADM
    sentiment_scores = getSentimentScores(news_headline)
    stock_data['subjectivity'] = sentiment_scores["Subjectivity"]
    stock_data['polarity'] = sentiment_scores["Polarity"]
    stock_data['compound'] = sentiment_scores["Compound"]
    stock_data['negative'] = sentiment_scores["Negative"]
    stock_data['neutral'] = sentiment_scores["Neutral"]
    stock_data['positive'] = sentiment_scores["Positive"]
    # Combine stock data and sentiment scores into a DataFrame
    today_data = stock_data.tail(num)
    today_data = today_data.drop(['Adj Close'], axis=1).values
    forecast_pred = model.predict(today_data)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {-1*i}')
        day += 1'''
    st.write('Predicted values in the upcoming days')
    st.line_chart(-1*forecast_pred)

if __name__ == '__main__':
    main()