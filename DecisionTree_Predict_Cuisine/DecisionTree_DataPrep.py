# Import Libraries
import streamlit as st
import pandas as pd
from DecisionTree_DataUnderstanding import df_train
from DecisionTree_DataUnderstanding import df_test

# Create set list for all ingredients
all_ingredients = set()
for ingredients in df_train['ingredients']:
    all_ingredients = all_ingredients | set(ingredients)
len(all_ingredients)

# Section 3 - Data Preparation
def app():
    st.title("Section 3 - Data Preparation")
    st.write("""
    ## **Data Checking & Cleaning** (If Necessary)
    """)
    st.subheader("Type of Cuisine Distribution")
    st.write(df_train['cuisine'].value_counts())
    st.write("""
    There are a total of {} unique type of cuisine in this dataset, and there are total of 6 type of
    cuisine that have significant difference than the other type of cuisine. Which are Italian, 
    Mexican, Southern US, Indian, Chinese and French (all of them are above 2000).
    """.format(len(df_train['cuisine'].unique())))

    st.write("""---""")

    st.subheader("Unique Ingredients")
    st.write(len(all_ingredients))
    st.write("""
    There are a total of {} unique ingredients in this dataset.
    """.format(len(all_ingredients)))

    st.write("""---""")

    st.subheader("Missing Values")
    st.write("**df_train**")
    st.write(df_train.isnull().sum())
    st.write("**df_test**")
    st.write(df_test.isnull().sum())

    st.write("""---""")

    st.write("""
    The datasets are cleaned and ready to be processed in the next phase which is Exploratory 
    Data Analysis
    """)


