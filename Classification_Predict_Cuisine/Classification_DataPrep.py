# Import Libraries
import streamlit as st
import re
import pandas as pd
from Classification_DataUnderstanding import df_train
from Classification_DataUnderstanding import df_test
from sklearn.preprocessing import MultiLabelBinarizer

# Create set list for all ingredients
all_ingredients = set()
for ingredients in df_train['ingredients']:
    all_ingredients = all_ingredients | set(ingredients)
len(all_ingredients)

# Create a copy of df_train and df_test for String Handling & One-Hot Encoding method
df_train_copy = df_train.copy()
df_test_copy = df_test.copy()


# Define function for string manipulation / string handling
@st.cache
def str_handling(data, column):
    list_of_lists = []
    for row in data[column]:
        list_of = []
        for lists in row:
            lists = re.sub(r"(\d)", "", lists)  # remove digits
            lists = re.sub(r"\([^)]*\)", "", lists)  # remove parentheses and its content
            lists = re.sub(u"\u2122", "", lists)  # remove trademark char
            lists = re.sub(r"[^\x00-\x7F]+", "", lists)  # remove unicode char
            lists = re.sub(r"%\s", "", lists)  # remove percentage sign
            lists = re.sub(r"/ to  lb.", "", lists)  # remove random words
            lists = lists.strip()  # remove leading and trailing whitespace
            lists = lists.lower()  # convert to lowercase
            list_of.append(lists)
        list_of_lists.append(list_of)
    data[column] = list_of_lists


# Define function for One-Hot Encoding Method
@st.cache
def one_hot(data, column):
    # Import necessary libraries
    # import pandas as pd
    # from sklearn.preprocessing import MultiLabelBinarizer

    # Create object for MultiLabelBinarizer
    mlb = MultiLabelBinarizer(sparse_output=True)

    return data.join(pd.DataFrame.sparse.from_spmatrix(mlb.fit_transform(data.pop(column)),
                                                       index=data.index,
                                                       columns=mlb.classes_))


# ---------- TRAIN DATASET ----------
# String Handling
str_handling(data=df_train_copy, column='ingredients')

# Apply One-Hot Encoding method to clean dataframe
df_train_onehot = one_hot(data=df_train_copy, column='ingredients')

# ---------- TEST DATASET ----------
# String Handling
str_handling(data=df_test_copy, column='ingredients')

# Apply One-Hot Encoding method to clean dataframe
df_test_onehot = one_hot(data=df_test_copy, column='ingredients')


# Section 3 - Data Preparation
@st.cache
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
