from Tweets import Tweets
from Prediction import Prediction
import streamlit as st
import pandas as pd

prediction = Prediction()
tweets = Tweets()

st.title("Welcome to :blue[Stock Price Predictor] Web App!")

stock = st.text_input("Enter stock symbol: ")
with st.spinner("Please Wait..."):
    prediction.fetchData(stock, "max")

    st.header(f"OHLC Graphs of {stock}")
    graph = st.radio(
        label="Enter type of graph: ",
        options=["Open", "High", "Low", "Close", "Volume"],
        horizontal=True
    )

    if(graph == "Volume"):
        prediction.plotGraph("Volume")
    elif(graph == "Open"):
        prediction.plotGraph("Open")
    elif(graph == "Close"):
        prediction.plotGraph("Close")
    elif(graph == "High"):
        prediction.plotGraph("High")
    elif(graph == "Low"):
        prediction.plotGraph("Low")

    st.header("Moving Averages...")
    col1, col2, col3 = st.columns(3)
    col1.subheader("10 Day MA")
    prediction.plotMA(10, col1, "#FF0000")
    col2.subheader("20 Day MA")
    prediction.plotMA(20, col2, "#FFFF00")
    col3.subheader("50 Day MA")
    prediction.plotMA(50, col3, "#00FF00")
    st.subheader("Superimposed Moving Averages")
    prediction.plotMAinOne()

    st.header("Predictions...")
    st.subheader("Based on historical data ")
with st.spinner("Please Wait..."):
    #prediction.predictPrices()
    prediction.testPredictionModel()
    prediction.plotPredictedPrices()
