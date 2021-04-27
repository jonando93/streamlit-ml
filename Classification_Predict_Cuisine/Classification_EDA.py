# Import Libraries
import pandas as pd
import streamlit as st
from Classification_DataPrep import df_train
from collections import Counter

# Create variable for all entries (number of rows)
row_entries = len(df_train['id'])

# Create 2 variables for total_ingredients and average_ingredients
total_ingredients = 0
for ingredients in df_train['ingredients']:
    total_ingredients += len(ingredients)
average_ingredients = total_ingredients / row_entries

# Create variable for cuisine distributions
cuisine_dist = df_train['cuisine'].value_counts()

# Create variable for ingredients per recipe and least ingredients
ingredients_per_recipe = []
for ingredients in df_train['ingredients']:
    ingredients_per_recipe.append(len(ingredients))
least_ingredients = min(ingredients_per_recipe)
most_ingredients = max(ingredients_per_recipe)

# Create new dataframe to save ingredients per recipe distribution
num_of_ing_dist = Counter(ingredients_per_recipe)
df_num_of_ing_dist = pd.DataFrame.from_dict(num_of_ing_dist, orient='index', columns=['occurrences'])
df_num_of_ing_dist.index.name = 'num_of_ingredients'


# Section 4 - Exploratory Data Analysis
def app():
    st.title("Section 4 - Exploratory Data Analysis")
    st.write("""
    In Data Preparation section, we have create a data regarding type of cuisine distribution. 
    In this Exploratory Data Analysis phase, I am going to visualize the data.
    """)
    st.subheader("Type of Cuisine Distribution")
    st.bar_chart(cuisine_dist)

    st.write("""---""")

    st.subheader("Average Amount of Ingredients per Recipe")
    st.code("total_ingredients = 0\n"
            "for ingredients in df_train['ingredients']:\n"
            "     total_ingredients += len(ingredients)\n"
            "average_ingredients = total_ingredients / row_entries\n"
            "print(average_ingredients)")
    st.write(average_ingredients)
    st.write("""
    The average amount of ingredients per recipe is between 10 and 11
    """)

    st.write("""---""")

    st.subheader("Least Amount of Ingredient(s) in All Recipes")
    st.write(least_ingredients)

    st.subheader("Most Amount of Ingredients in All Recipes")
    st.write(most_ingredients)

    st.write("""---""")

    st.subheader("Amount of Ingredients Distribution")
    st.write(df_num_of_ing_dist)
    st.bar_chart(df_num_of_ing_dist)

    st.write("""---""")
