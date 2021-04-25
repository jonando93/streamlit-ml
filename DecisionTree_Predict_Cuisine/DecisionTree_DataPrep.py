# Import Libraries
import streamlit as st
import re
import pandas as pd
from DecisionTree_DataUnderstanding import df_train
from DecisionTree_DataUnderstanding import df_test

# Create set list for all ingredients
all_ingredients = set()
for ingredients in df_train['ingredients']:
    all_ingredients = all_ingredients | set(ingredients)
len(all_ingredients)

# Removing any unnecessary string characters
list_of_lists = []
for row in df_train['ingredients']:
    l = []
    for lists in row:
        # Remove Digits
        lists = re.sub(r"(\d)", "", lists)

        # Remove Content Inside Parentheses
        lists = re.sub(r"\([^)]*\)", "", lists)

        # Remove TradeMark Char
        lists = re.sub(u"\u2122", "", lists)

        # Remove Unicode Char
        lists = re.sub(r"[^\x00-\x7F]+", "", lists)

        # Remove percentage sign
        lists = lists.strip('%')

        # Remove random words
        lists = lists.strip("/ to lb.")
        lists = re.sub(r"-bone", "bone", lists)

        # Remove leading and trailing whitespace
        lists = lists.strip()

        # Convert to lowercase
        lists = lists.lower()

        l.append(lists)
    list_of_lists.append(l)

df_train['ingredients'] = list_of_lists

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

    st.subheader("Replacing String Characters")
    st.write("""
    While checking the dataset, I noticed that some of the ingredients include string characters
    such as '(', ')', '%' and '.'. For example, '(10 oz.) tomato sauce' should be labeled only as
    'tomato sauce', and also '1% low-fat butter' should only be labeled as 'low-fat butter'.
    """)
    st.code("re.sub(r'(\d)', '', list) # remove digits \n"
            "re.sub(r'\([^)]*\), '', list) # remove parentheses and its content \n"
            "re.sub(r'[^\x00-\x7F]+', '', list) # remove unicode characters")

    st.write("""---""")

    st.write("""
    The datasets are cleaned and ready to be processed in the next phase which is Exploratory 
    Data Analysis
    """)
