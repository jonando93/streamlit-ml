# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
from DecisionTree_DataPrep import df_train
from DecisionTree_DataPrep import df_train_onehot
from DecisionTree_DataPrep import df_test_onehot
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import graphviz
from sklearn.tree import export_graphviz

# ----- TEST DATASET -----
# Since the TEST DATASET does not include cuisine column, I will use TRAIN DATASET instead.
ingredients_test = df_test_onehot.iloc[:, 1:]

# ----- TRAIN DATASET -----
# I will split the Train dataset to create 4 variables (xy train & xy test)

# Create 2 new variables for cuisine and ingredients (x & y)
cuisines = df_train_onehot['cuisine'] # y
ingredients = df_train_onehot.iloc[:, 2:] # x

x_train, x_test, y_train, y_test = train_test_split(ingredients, cuisines, test_size=0.2, random_state=69)

# Create object for decision tree model called mango_tree
mango_tree = tree.DecisionTreeClassifier(class_weight=None,
                                         criterion='gini',
                                         max_depth=None,
                                         max_features=None,
                                         max_leaf_nodes=None,
                                         min_impurity_decrease=0.0,
                                         min_impurity_split=None,
                                         min_samples_leaf=1,
                                         min_samples_split=2,
                                         min_weight_fraction_leaf=0.0,
                                         random_state=None,
                                         splitter='best')
mango_tree.fit(x_train, y_train)
y_pred = mango_tree.predict(x_test)
print("Accuracy is ", accuracy_score(y_pred, y_test)*100)

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

