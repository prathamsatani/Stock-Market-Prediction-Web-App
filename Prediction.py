import pandas as pd
import yfinance as yf
import numpy as np
import plotly as pl
import streamlit as st
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

class Prediction:
    def __init__(self) -> None:
        pass

    def fetchData(self, name: str, period: str) -> None:
        self.stock = yf.Ticker(name).history(period)
    
    def plotGraph(self, type):
        if(type == "Volume"):
            st.line_chart(data=self.stock["Volume"])
        elif(type == "Close"):
            st.line_chart(data=self.stock["Close"])
        elif(type == "Open"):
            st.line_chart(data=self.stock["Open"])
        elif(type == "High"):
            st.line_chart(data=self.stock["High"])
        elif(type == "Low"):
            st.line_chart(data=self.stock["Low"])

    def plotMA(self, period, column, d_color):
        values = self.stock["Close"].rolling(period).mean()
        column.line_chart(data=values, color=d_color)
    
    def plotMAinOne(self):
        period = [10, 20, 30]
        ma = [] 
        for p in period:
            ma.append(self.stock["Close"].rolling(p).mean())
        
        dict = {"10-Day Moving Average":ma[0], "20-Day Moving Average": ma[1], "50-Day Moving Average": ma[2]}
        df = pd.DataFrame(dict)
        
        st.line_chart(dict, color=["#FF0000", "#FFFF00", "#00FF00"])

    def predictionModel(self, shape):
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=shape))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        return model
    
    def createDataset(self):
        self.df = self.stock
        self.data = self.df.filter(['Close'])
        self.dataset = self.data.values
        self.training_data_len = int(np.ceil( len(self.dataset) * .95 ))
        return self.dataset
    
    def predictPrices(self):
        self.createDataset()
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.scaled_data = self.scaler.fit_transform(self.dataset)
        training_data_len = int(np.ceil( len(self.dataset) * .95 ))

        train_data = self.scaled_data[0:int(training_data_len), :]

        x_train = []
        y_train = []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
            if i<= 61:
                print(x_train)
                print(y_train)
                print()

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        self.model = self.predictionModel((x_train.shape[1], 1))

        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics="accuracy")

        self.model.fit(x_train, y_train, batch_size=1, epochs=1)

    def testPredictionModel(self):
        test_data = self.scaled_data[self.training_data_len - 60: , :]
        x_test = []
        y_test = self.dataset[self.training_data_len:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])

        x_test = np.array(x_test)

        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

        self.predictions = self.model.predict(x_test)
        self.predictions = self.scaler.inverse_transform(self.predictions)

        return self.predictions
    
    def plotPredictedPrices(self):
        train = self.data[:self.training_data_len]
        valid = self.data[self.training_data_len:]
        valid['Predictions'] = self.predictions
        dict = {"train": train["Close"], "validation":valid["Close"], "prediction":valid["Predictions"]}
        st.line_chart(dict, color=["#0000FF", "#FFFF00", "#00FF00"])











P1 = Prediction()
P1.fetchData("IOC.NS", "10y")
P1.createDataset()