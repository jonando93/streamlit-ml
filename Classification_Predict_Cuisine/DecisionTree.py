# Import Libraries
import DecisionTree_Intro
import DecisionTree_DataUnderstanding
import DecisionTree_DataPrep
import DecisionTree_EDA
import DecisionTree_MLModel
import DecisionTree_ModelShowcase
import streamlit as st

# Create Object for Navigation Page Called PAGES
PAGES = {
    "Introduction": DecisionTree_Intro,
    "Data Understanding": DecisionTree_DataUnderstanding,
    "Data Preparation": DecisionTree_DataPrep,
    "Exploratory Data Analysis": DecisionTree_EDA,
    "Machine Learning Model": DecisionTree_MLModel,
    "Machine Learning Model Showcase": DecisionTree_ModelShowcase
}

# Create the Sidebar for Navigation Page
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()

# Use this line in command prompt or git bash or any other terminal
# $ streamlit run DecisionTree.py