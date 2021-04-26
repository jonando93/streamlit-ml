# Import Libraries
import streamlit as st
import re
import pandas as pd
from Classification_DataUnderstanding import df_train
from Classification_DataUnderstanding import df_test
from sklearn.preprocessing import MultiLabelBinarizer

# ----- TRAIN DATASET -----
# Removing any unnecessary string characters
list_of_lists1 = []
for row1 in df_train['ingredients']:
    l1 = []
    for lists1 in row1:
        # Remove Digits
        lists1 = re.sub(r"(\d)", "", lists1)

        # Remove Content Inside Parentheses
        lists1 = re.sub(r"\([^)]*\)", "", lists1)

        # Remove TradeMark Char
        lists1 = re.sub(u"\u2122", "", lists1)

        # Remove Unicode Char
        lists1 = re.sub(r"[^\x00-\x7F]+", "", lists1)

        # Remove percentage sign
        lists1 = re.sub(r"%\s", "", lists1)

        # Remove random words
        lists1 = re.sub(r"/ to  lb.", "", lists1)

        # Remove leading and trailing whitespace
        lists1 = lists1.strip()

        # Convert to lowercase
        lists1 = lists1.lower()

        l1.append(lists1)
    list_of_lists1.append(l1)

df_train['ingredients'] = list_of_lists1

# Create set list for all ingredients
all_ingredients = set()
for ingredients in df_train['ingredients']:
    all_ingredients = all_ingredients | set(ingredients)
len(all_ingredients)

# Create a copy of df_train for One-Hot Encoding method
df_train_copy = df_train.copy()

# Create object for MultiLabelBinarizer
mlb = MultiLabelBinarizer(sparse_output=True)

# Create new DataFrame for One-Hot Encoding method
df_train_onehot = df_train_copy.join(
    pd.DataFrame.sparse.from_spmatrix(
        mlb.fit_transform(df_train_copy.pop('ingredients')),
        index=df_train_copy.index,
        columns=mlb.classes_))

# ----- TEST DATASET -----
# Removing any unnecessary string characters
list_of_lists2 = []
for row2 in df_test['ingredients']:
    l2 = []
    for lists2 in row2:
        # Remove Digits
        lists2 = re.sub(r"(\d)", "", lists2)

        # Remove Content Inside Parentheses
        lists2 = re.sub(r"\([^)]*\)", "", lists2)

        # Remove TradeMark Char
        lists2 = re.sub(u"\u2122", "", lists2)

        # Remove Unicode Char
        lists2 = re.sub(r"[^\x00-\x7F]+", "", lists2)

        # Remove percentage sign
        lists2 = re.sub(r"%\s", "", lists2)

        # Remove random words
        lists2 = re.sub(r"/ to  lb.", "", lists2)

        # Remove leading and trailing whitespace
        lists2 = lists2.strip()

        # Convert to lowercase
        lists2 = lists2.lower()

        l2.append(lists2)
    list_of_lists2.append(l2)

df_test['ingredients'] = list_of_lists2

# Create a copy of df_train for One-Hot Encoding method
df_test_copy = df_test.copy()

# Create new DataFrame for One-Hot Encoding method
df_test_onehot = df_test_copy.join(
    pd.DataFrame.sparse.from_spmatrix(
        mlb.fit_transform(df_test_copy.pop('ingredients')),
        index=df_test_copy.index,
        columns=mlb.classes_))

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
            "re.sub(r'[^\x00-\x7F]+', '', list) # remove unicode characters \n"
            "re.sub(r'%\s', '', list) # remove percentage sign and its trailing whitespace \n"
            "list.strip() # remove leading and trailing whitespace")

    st.write("""---""")

    st.write("""
    The datasets are cleaned and ready to be processed in the next phase which is Exploratory 
    Data Analysis
    """)
