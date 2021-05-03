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
print(input_x_test)


# Section 6 - Machine Learning Model Showcase
def app():
    st.title("Section 6 - ML Model Showcase")
    st.write("""
    In this section, I am going to showcase all the ML models that have been build in the 
    previous section, using the **x_test** dataset that are provided to predict the type of 
    cuisine.
    
    In the train dataset, there is a Greek cuisine that consists of:
    - Kosher Salt
    - Garlic
    - Greek Yogurt
    - Cracked Black Pepper
    - English Cucumber
    - Shallots
    - Dill
    - White Vinegar
    - Extra-Virgin Olive Oil
    - Fresh Lemon Juice
    
    Or, Russian cuisine that consists of:
    - Bread Crumb Fresh
    - Vegetable Oil
    - Sour Cream
    - Unsalted Butter
    - All-Purpose Flour
    - Water
    - Salt
    - Flat Leaf Parsley
    - Large Eggs
    - Farmer Cheese
    
    Let's try to input one of these ingredients to see how well these Machine Learning models 
    predict type of cuisine!
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
    select_ing6 = st.selectbox("Input 6th Ingredients:", input_x_test.columns)
    select_ing7 = st.selectbox("Input 7th Ingredients:", input_x_test.columns)
    select_ing8 = st.selectbox("Input 8th Ingredients:", input_x_test.columns)
    select_ing9 = st.selectbox("Input 9th Ingredients:", input_x_test.columns)
    select_ing10 = st.selectbox("Input 10th Ingredients:", input_x_test.columns)

    # Create 'Predict' button to predict cuisine based on ingredients
    if st.button("Predict"):
        if model == "Decision Tree":
            input_x_test[select_ing1] = 1
            input_x_test[select_ing2] = 1
            input_x_test[select_ing3] = 1
            input_x_test[select_ing4] = 1
            input_x_test[select_ing5] = 1
            input_x_test[select_ing6] = 1
            input_x_test[select_ing7] = 1
            input_x_test[select_ing8] = 1
            input_x_test[select_ing9] = 1
            input_x_test[select_ing10] = 1
            st.write(dec_tree_model(xtrain=x_train,
                                    ytrain=y_train,
                                    xtest=input_x_test)[0])
        elif model == "Logistic Regression":
            input_x_test[select_ing1] = 1
            input_x_test[select_ing2] = 1
            input_x_test[select_ing3] = 1
            input_x_test[select_ing4] = 1
            input_x_test[select_ing5] = 1
            input_x_test[select_ing6] = 1
            input_x_test[select_ing7] = 1
            input_x_test[select_ing8] = 1
            input_x_test[select_ing9] = 1
            input_x_test[select_ing10] = 1
            st.write(log_reg_model(xtrain=x_train,
                                   ytrain=y_train,
                                   xtest=input_x_test)[0])
        else:
            input_x_test[select_ing1] = 1
            input_x_test[select_ing2] = 1
            input_x_test[select_ing3] = 1
            input_x_test[select_ing4] = 1
            input_x_test[select_ing5] = 1
            input_x_test[select_ing6] = 1
            input_x_test[select_ing7] = 1
            input_x_test[select_ing8] = 1
            input_x_test[select_ing9] = 1
            input_x_test[select_ing10] = 1
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
        input_x_test[select_ing6] = 0
        input_x_test[select_ing7] = 0
        input_x_test[select_ing8] = 0
        input_x_test[select_ing9] = 0
        input_x_test[select_ing10] = 0
        st.write("""
        Input has been reset successfully! \n
        Try another combination :D
        """)

    st.write("""---""")

    st.subheader("Discussion & Conclusion")
    st.write("""
    For the first set of ingredients, both Decision Tree and Logistic Regression predicts correctly,
    while the Random Forest model predicts French.
    
    But for the second set of ingredients, all of the Machine Learning models successfully predict
    Russian.
    
    These 3 models have the accuracy of 80 (Decision Tree), 83 (Log. Regression) and 84 (Random 
    Forest).
    """)
