# Import Libraries
import Classification_Intro
import Classification_DataUnderstanding
import Classification_DataPrep
import Classification_EDA
import Classification_MLModel
import Classification_ModelShowcase
import streamlit as st

# Create Object for Navigation Page Called PAGES
PAGES = {
    "Introduction": Classification_Intro,
    "Data Understanding": Classification_DataUnderstanding,
    "Data Preparation": Classification_DataPrep,
    "Exploratory Data Analysis": Classification_EDA,
    "Machine Learning Model": Classification_MLModel,
    "Machine Learning Model Showcase": Classification_ModelShowcase
}

# Create the Sidebar for Navigation Page
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()

# Use this line in command prompt or git bash or any other terminal
# $ streamlit run Classification.py