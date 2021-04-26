# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
from DecisionTree_DataPrep import df_train
from DecisionTree_DataPrep import df_train_onehot
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
import graphviz
from sklearn.tree import export_graphviz

# Create 2 new variables for cuisine and ingredients
cuisines = df_train_onehot['cuisine'] # y_train
ingredients = df_train_onehot.iloc[:, 2:] # x_train

# Adjust width of the Decision Tree model (and print its accuracy)
for n in range(4,30):
    depth = n+1

    # Create object for decision tree model called mango_tree
    mango_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=depth)
    mango_tree.fit(ingredients, cuisines)
    pred_cuisines = mango_tree.predict(ingredients)
    print("Accuracy is ", accuracy_score(pred_cuisines, cuisines)*100, "% for Depth:", depth)


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

    st.write("""---""")

