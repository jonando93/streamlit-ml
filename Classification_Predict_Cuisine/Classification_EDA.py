# Import Libraries
import pandas as pd
import numpy as np
import streamlit as st
from Classification_DataPrep import df_train_copy
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

# Create a copy of the clean dataset
df_train_clean = df_train_copy.copy()

# Create variable for all entries (number of rows)
row_entries = len(df_train_clean['id'])

# Create 2 variables for total_ingredients and average_ingredients
total_ingredients_before = 0
for ingredients in df_train_clean['ingredients']:
    total_ingredients_before += len(ingredients)
average_ingredients_before = total_ingredients_before / row_entries

# Create variable for cuisine distributions
cuisine_dist_before = df_train_clean['cuisine'].value_counts()

# Create variable for ingredients per recipe and least ingredients
ingredients_per_recipe = []
for ingredients in df_train_clean['ingredients']:
    ingredients_per_recipe.append(len(ingredients))
least_ingredients = min(ingredients_per_recipe)
most_ingredients = max(ingredients_per_recipe)

# Create new dataframe to save ingredients per recipe distribution
num_of_ing_dist = Counter(ingredients_per_recipe)
df_num_of_ing_dist = pd.DataFrame.from_dict(num_of_ing_dist, orient='index', columns=['occurrences'])
df_num_of_ing_dist.index.name = 'num_of_ingredients'

# Remove recipes that have amount of ingredients less than 4 and above 17
df_train_clean['len'] = df_train_clean['ingredients'].str.len()  # create new column to calc. len()
df_train_clean = df_train_clean[df_train_clean['len'] > 3]  # remove recipes that have less than 4 ing.
df_train_clean = df_train_clean[df_train_clean['len'] < 18]  # remove recipes that have more than 17 ing.
df_train_clean = df_train_clean.reset_index(drop=True)  # reset index


# Define a function for undersampling type of cuisine (class)
def undersampling(data, value, column='cuisine', threshold=2000):
    data_indices = data[data[column] == value].index
    np.random.seed(69)
    random_indices = np.random.choice(data_indices, threshold, replace=False)
    return data.loc[random_indices]  # return a dataframe


# Define a function for oversampling type of cuisine (class)
def oversampling(data, value, column='cuisine'):
    init_data = data[data[column] == value]
    data_copy = data[data[column] == value].copy()
    init_data = pd.concat([init_data, data_copy])  # return a dataframe
    return init_data


def resampling(dataset, column='cuisine'):
    appended_data = []
    cuisine_list = dataset[column].unique().tolist()
    for lists in cuisine_list:
        filter_value = dataset[dataset[column] == lists]
        if len(filter_value[column]) < 1000:
            oversampled = oversampling(data=dataset, value=lists)
            appended_data.append(oversampled)
        elif len(filter_value[column]) > 2000:
            undersampled = undersampling(data=dataset, value=lists)
            appended_data.append(undersampled)
        else:
            sampled = filter_value
            appended_data.append(sampled)
    return pd.concat(appended_data)


# Resample the clean dataset
df_train_resampled = resampling(dataset=df_train_clean)

# Drop 'len' column before applying one-hot encoding
df_train_resampled = df_train_resampled.drop(['len'], axis=1)

# Create variable for Amount of Ingredients Distribution after Adjustment

# Create variable for Cuisine Distribution after Resampling
cuisine_dist_after = df_train_resampled['cuisine'].value_counts()


# Define function for One-Hot Encoding Method
@st.cache(suppress_st_warning=True)
def one_hot(data, column):
    # Import necessary libraries
    # import pandas as pd
    # from sklearn.preprocessing import MultiLabelBinarizer

    # Create object for MultiLabelBinarizer
    mlb = MultiLabelBinarizer(sparse_output=True)

    return data.join(pd.DataFrame.sparse.from_spmatrix(mlb.fit_transform(data.pop(column)),
                                                       index=data.index,
                                                       columns=mlb.classes_))


# Apply One-Hot Encoding method to clean dataframe
df_train_onehot = one_hot(data=df_train_resampled, column='ingredients')


# Section 4 - Exploratory Data Analysis
def app():
    st.title("Section 4 - Exploratory Data Analysis")
    st.write("""
    In Data Preparation section, we have create a data regarding type of cuisine distribution. 
    In this Exploratory Data Analysis phase, I am going to visualize the data.
    """)
    st.subheader("Type of Cuisine Distribution")
    st.bar_chart(cuisine_dist_before)

    st.write("""---""")

    st.subheader("Average Amount of Ingredients per Recipe")
    st.code("total_ingredients = 0\n"
            "for ingredients in df_train['ingredients']:\n"
            "     total_ingredients += len(ingredients)\n"
            "average_ingredients = total_ingredients / row_entries\n"
            "print(average_ingredients)")
    st.write(average_ingredients_before)
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

    st.write("### **Adjusting / Normalizing Amount of Ingredients Distribution**")
    st.write("""
    Based on the plot in 'Amount of Ingredients Distribution' above, there are a lot of recipes that
    have less than 4 or more than 17 ingredients and have the number of occurrences below 1000.
    
    The majority of data falls between 4 and 19 ingredients per recipe, and these recipes
    occurred more than 1000. Since the classification model such as Decision Tree will treat
    minority as outlier or noise, I will remove these data so it wont affect the performance
    of the classification models.
    """)

    st.write("### **Random Resampling an Imbalanced Dataset**")
    st.write("""
    As shown in the 'Type of Cuisine Distribution', we can clearly see that this dataset is
    imbalanced in terms of cuisine (class) distribution. To solve this problem, I have
    to apply random resampling to the dataset.
    
    > The two main approaches to randomly resampling an imbalanced dataset are to delete
    > examples from the majority class, called **undersampling**, and to duplicate examples 
    > from the minority class, called **oversampling**.
    > - [MachineLearningMastery](https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/)
    
    Based on the plot shown in the 'Type of Cuisine Distribution', I will set the threshold
    value of 2000. I will apply oversampling/undersampling for each cuisine to meet the
    threshold value.
    """)

    st.write("""---""")

    st.write("### **After Dataset Adjustment and Resampling**")
    st.subheader("Type of Cuisine Distribution")
    st.bar_chart(cuisine_dist_after)
    st.write("As seen on the bar chart above, the type of cuisine distribution is more balanced.")
