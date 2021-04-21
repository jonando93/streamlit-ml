# Import Libraries
import streamlit as st

# Section 1 - Introduction & Idea
def app():
    st.title("Section 1 - Introduction & Idea")
    st.write("""
    ## **What is Machine Learning?**
    > Machine Learning is a method of data analysis that automates analytical model building. It is a 
    > branch of artificial intelligence based on the idea that systems can learn from data, identify 
    > patterns and make decisions with minimal human intervention.
    >
    > -- Quoted from [SAS](https://www.sas.com/en_id/insights/analytics/machine-learning.html)
    
    ## **Types of Machine Learning**
    There are multiple types of machine learning, but the most commonly known types of machine learning are
    supervised and unsupervised learning model.
    
    > In a **supervised learning** model, the algorithm learns on a labeled dataset, providing an answer
    > key that the algorithm can use to evaluate its accuracy on training data. An **unsupervised learning**
    > model, in contrast, provides unlabeled data that the algorithm tries to make sense of by extracting 
    > features and pattern on its own.
    >
    > -- Quoted from [NVidia](https://blogs.nvidia.com/blog/2018/08/02/supervised-unsupervised-learning/)
    
    There are two main areas where supervised learning is useful: classification problems and regression
    problems
    
    ## **Project Idea**
    In this project, I am going to try to build one of the Machine Learning model called **Decision Tree**
    to predict the origin of cuisine based on only ingredients. Decision Tree falls under the classification
    category which is belong to the supervised learning model
    
    ***
    """)