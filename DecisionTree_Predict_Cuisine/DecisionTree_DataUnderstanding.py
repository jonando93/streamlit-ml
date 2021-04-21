# Import Libraries
import pandas as pd
import streamlit as st
import json

# Open JSON file
JSON_data_train = open('Cuisine_train.json')
JSON_data_test = open('Cuisine_test.json')

# Return JSON object as dict
data_train = json.load(JSON_data_train)
data_test = json.load(JSON_data_test)

# Convert the dict into 2 new dataframe called df_train and df_test
df_train = pd.DataFrame(data_train)
df_test = pd.DataFrame(data_test)

print(df_train.dtypes)

# Section 2 - Data Understanding
def app():
    st.title("Section 2 - Data Understanding") # Add title
    st.write("""
    ## **Content**
    > In this dataset, we include the recipe id, the type of cuisine, and the list of ingredients of each 
    > recipe (of variable length). The data are stored in JSON format.
    > - **train.json** - the training set containing recipes id, type of cuisine, and list of ingredients
    > - **test.json** - the test set containing recipes id and list of ingredients.
    >
    > -- via [Kaggle](https://www.kaggle.com/kaggle/recipe-ingredients-dataset)
    """)

    st.write("""
    ## **Column Names**
    ### train.json
    - **id** - recipe id, or unique id for each recipe.
    - **cuisine** - type of cuisine (origin).
    - **ingredients** - list of ingredients inside the cuisine.
    """)
    # Show df_train.head()
    st.dataframe(df_train.head())

    st.write("""
    ### test.json
    - **id** - recipe id, or unique id for each recipe.
    - **ingredients** - list of ingredients inside the cuisine.
    """)
    # Show df_test.head()
    st.dataframe(df_test.head())

    st.write("""
    ## **Column Data Types**
    - **id** - int64
    - **cuisine** - object
    - **ingredients** - object
    """)

