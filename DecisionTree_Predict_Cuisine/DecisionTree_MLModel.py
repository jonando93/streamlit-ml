# Import Libraries
import streamlit as st
import pandas as pd
from DecisionTree_DataPrep import df_train
from DecisionTree_DataPrep import df_train_onehot

# Section 5 - Machine Learning Model
def app():
    st.title("Section 5 - Machine Learning Model")
    st.write("""
    In this section, I am going to build the Decision Tree Model to predict cuisine based on 
    ingredients. In the previous section, I have prepared the dataset that will be used in 
    building the model. 
    """)

    st.subheader("One-Hot Encoding")
    st.write("""
    In our dataset, the 'ingredients' column are in list format.
    """)
    st.write(df_train['ingredients'].head(3))
    st.write("""
    I will use **one-hot encoding** method to make separate columns for each ingredients.
    """) # One-Hot Dataframe already prepared in DataPrep section
    st.write(df_train_onehot.head(3))