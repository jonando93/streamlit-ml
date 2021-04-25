# Import Libraries
import streamlit as st
import pandas as pd
from DecisionTree_DataPrep import df_train
from sklearn.preprocessing import MultiLabelBinarizer

df_train_copy = df_train.copy()

# Create object for MultiLabelBinarizer
mlb = MultiLabelBinarizer(sparse_output=True)

# Create new DataFrame for One-Hot Encoding method
df_train_onehot = df_train_copy.join(
    pd.DataFrame.sparse.from_spmatrix(
        mlb.fit_transform(df_train_copy.pop('ingredients')),
        index=df_train_copy.index,
        columns=mlb.classes_))
print(df_train_onehot)
print(df_train['ingredients'].head())
print(df_train_copy)

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
    """)
    st.write(df_train_onehot.head(3))