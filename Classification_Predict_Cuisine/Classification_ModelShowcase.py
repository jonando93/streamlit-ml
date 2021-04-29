# Import Libraries
import streamlit as st
from Classification_MLModel import x_train
from Classification_MLModel import y_train
from Classification_MLModel import x_test
from Classification_MLModel import dec_tree_model
from Classification_MLModel import log_reg_model
from Classification_MLModel import rand_forest_model


# Make a new dataframe that contains all the column names in x_test dataframe
# Change all the value to 0.
input_x_test = x_test.copy().head(1)
input_x_test.reset_index(drop=True, inplace=True)


# Section 6 - Machine Learning Model Showcase
def app():
    st.title("Section 6 - ML Model Showcase")
    st.write("""
    In this section, I am going to showcase all the ML models that have been build in the previous 
    section, using the **x_test** dataset that are provided to predict the type of cuisine.
    
    In the train dataset, there is a Mexican food that consists of:
    - Blanched Almond Flour
    - Sea Salt
    - Tapioca Flour
    - Warm Water
    - Mild Olive Oil
    
    Let's try to input these ingredients to see how well these Machine Learning models predict type 
    of cuisine!
    """)

    st.write("""---""")

    # Insert Select Box to select ML Model
    model = st.selectbox("Select Machine Learning Model:", ["Decision Tree",
                                                            "Logistic Regression",
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
        else:
            input_x_test[select_ing1] = 1
            input_x_test[select_ing2] = 1
            input_x_test[select_ing3] = 1
            input_x_test[select_ing4] = 1
            input_x_test[select_ing5] = 1
            st.write(rand_forest_model(xtrain=x_train,
                                       ytrain=y_train,
                                       xtest=input_x_test)[0])

    st.write("""---""")

    st.write('Please reset the input after each prediction using the reset button below.')

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

    st.write("""---""")

    st.subheader("Discussion & Conclusion")
    st.write("""
    Decision Tree model predicts Southern US, which is close to Mexico.
    
    Logistic Regression model also predicts Southern US. *This is our highest scoring model.*
    
    Random Forest model predicts Mexican. *The only model that predicts correctly.*
    
    Based on these results, some of the models almost predicted the correct result, maybe some of 
    the Southern US cuisines are influenced by Mexican Cuisine.
    
    Random Forest model is an improved version of Decision Tree model. Decision Tree model works 
    poorly compared to Random Forest model whenever there are a lot of labels (class) that needs
    to be distinguished. In this dataset, there are 20 type of cuisines.
    """)
