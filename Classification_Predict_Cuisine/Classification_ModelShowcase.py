# Import Libraries
import streamlit as st
from Classification_MLModel import x_train
from Classification_MLModel import y_train
from Classification_MLModel import x_test
from Classification_MLModel import dec_tree_model
from Classification_MLModel import log_reg_model
from Classification_MLModel import knn_model
from Classification_MLModel import rand_forest_model


# Make a new dataframe that contains all the column names in x_test dataframe
# Change all the value to 0.
input_x_test = x_test.copy().head(1)
input_x_test.reset_index(drop=True, inplace=True)


# Section 6 - Machine Learning Model Showcase
@st.cache
def app():
    st.title("Section 6 - ML Model Showcase")
    st.write("""
    In this section, I am going to showcase all the ML models that have been build in the previous 
    section, using the test dataset that are provided to predict the type of cuisine.
    """)

    # Insert Select Box to select ML Model
    model = st.selectbox("Select Machine Learning Model:", ["Decision Tree",
                                                            "Logistic Regression",
                                                            "K-Nearest Neighbors",
                                                            "Random Forest"])

    # Insert Select Box to select ingredients
    select_ing1 = st.selectbox("Input 1st Ingredients:", input_x_test.columns)
    select_ing2 = st.selectbox("Input 2nd Ingredients:", input_x_test.columns)
    select_ing3 = st.selectbox("Input 3rd Ingredients:", input_x_test.columns)
    select_ing4 = st.selectbox("Input 4th Ingredients:", input_x_test.columns)
    select_ing5 = st.selectbox("Input 5th Ingredients:", input_x_test.columns)

    # Create 'Predict' button to predict cuisine based on ingredients
    if st.button("Predict"):
        if model == "Decision Tree":
            input_x_test[select_ing1] = 1
            input_x_test[select_ing2] = 1
            input_x_test[select_ing3] = 1
            input_x_test[select_ing4] = 1
            input_x_test[select_ing5] = 1
            st.write(dec_tree_model(xtrain=x_train,
                                    ytrain=y_train,
                                    xtest=input_x_test)[0])
        elif model == "Logistic Regression":
            input_x_test[select_ing1] = 1
            input_x_test[select_ing2] = 1
            input_x_test[select_ing3] = 1
            input_x_test[select_ing4] = 1
            input_x_test[select_ing5] = 1
            st.write(log_reg_model(xtrain=x_train,
                                   ytrain=y_train,
                                   xtest=input_x_test)[0])
        elif model == "K-Nearest Neighbors":
            input_x_test[select_ing1] = 1
            input_x_test[select_ing2] = 1
            input_x_test[select_ing3] = 1
            input_x_test[select_ing4] = 1
            input_x_test[select_ing5] = 1
            st.write(knn_model(xtrain=x_train,
                               ytrain=y_train,
                               xtest=input_x_test)[0])
        else:
            input_x_test[select_ing1] = 1
            input_x_test[select_ing2] = 1
            input_x_test[select_ing3] = 1
            input_x_test[select_ing4] = 1
            input_x_test[select_ing5] = 1
            st.write(rand_forest_model(xtrain=x_train,
                                       ytrain=y_train,
                                       xtest=input_x_test)[0])

    # Create 'Reset Input' button
    if st.button("Reset Input"):
        input_x_test[select_ing1] = 0
        input_x_test[select_ing2] = 0
        input_x_test[select_ing3] = 0
        input_x_test[select_ing4] = 0
        input_x_test[select_ing5] = 0
        st.write("""
        Input has been reset successfully! \n
        Try another combination :D
        """)
