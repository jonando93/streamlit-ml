# Import Libraries
import streamlit as st
import pandas as pd
from Classification_DataPrep import df_train
from Classification_DataPrep import df_train_onehot
from Classification_DataPrep import df_test_onehot
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ---------- TEST DATASET ----------
# Since the TEST DATASET does not include cuisine column, I will use TRAIN DATASET instead.
ingredients_test = df_test_onehot.iloc[:, 1:]

# ---------- TRAIN DATASET ----------
# I will split the Train dataset to create 4 variables (xy train & xy test)

# Create 2 new variables for cuisine and ingredients (x & y)
cuisines = df_train_onehot['cuisine']  # y
ingredients = df_train_onehot.iloc[:, 2:]  # x

x_train, x_test, y_train, y_test = train_test_split(ingredients, cuisines, test_size=0.2, random_state=69)


# MODEL 1 - DECISION TREE
@st.cache(suppress_st_warning=True)
def dec_tree_model(xtrain, ytrain, xtest):
    # Import necessary libraries
    # from sklearn.tree import DecisionTreeClassifier

    # Create callable object for DecisionTreeClassifier
    dec_tree = DecisionTreeClassifier(class_weight=None,
                                      criterion='gini',
                                      max_depth=None,
                                      max_features=None,
                                      max_leaf_nodes=None,
                                      min_impurity_decrease=0.0,
                                      min_impurity_split=None,
                                      min_samples_leaf=1,
                                      min_samples_split=2,
                                      min_weight_fraction_leaf=0.0,
                                      random_state=69,
                                      splitter='best')
    dec_tree.fit(xtrain, ytrain)  # train the model (fit x_train and y_train)
    return dec_tree.predict(xtest)  # Prediction


# Predict using the Decision Tree Model
yhat_dec_tree = dec_tree_model(xtrain=x_train, ytrain=y_train, xtest=x_test)
# Calculate the Decision Tree model accuracy
dec_tree_accuracy = accuracy_score(yhat_dec_tree, y_test)


# MODEL 2 - LOGISTIC REGRESSION
@st.cache(suppress_st_warning=True)
def log_reg_model(xtrain, ytrain, xtest):
    # Import necessary libraries
    # from sklearn.linear_model import LogisticRegression

    # Create callable object for LogisticRegression
    log_reg = LogisticRegression(C=1.0,
                                 class_weight=None,
                                 dual=False,
                                 fit_intercept=True,
                                 intercept_scaling=1,
                                 l1_ratio=None,
                                 multi_class='auto',
                                 n_jobs=None,
                                 penalty='l2',
                                 random_state=69,
                                 solver='liblinear',
                                 tol=0.0001,
                                 verbose=0,
                                 warm_start=False)
    log_reg.fit(xtrain, ytrain)
    return log_reg.predict(xtest)


# Predict using the Logistic Regression Model
yhat_log_reg = log_reg_model(xtrain=x_train, ytrain=y_train, xtest=x_test)
# Calculate the Logistic Regression model accuracy
log_reg_accuracy = accuracy_score(yhat_log_reg, y_test)


# MODEL 3 - K-NEAREST NEIGHBORS
@st.cache(suppress_st_warning=True)
def knn_model(xtrain, ytrain, xtest):
    # Import necessary libraries
    # from sklearn.neighbors import KNeighborsClassifier

    # Create callable object for KNeighborsClassifier
    knn = KNeighborsClassifier(algorithm='auto',
                               leaf_size=30,
                               metric='minkowski',
                               metric_params=None,
                               n_jobs=None,
                               n_neighbors=20,
                               p=2,
                               weights='uniform')
    knn.fit(xtrain, ytrain)
    return knn.predict(xtest)


# Predict using the K-Nearest Neighbors Model
yhat_knn = knn_model(xtrain=x_train, ytrain=y_train, xtest=x_test)
# Calculate the K-Nearest Neighbors model accuracy
knn_accuracy = accuracy_score(yhat_knn, y_test)


# MODEL 4 - RANDOM FOREST
@st.cache(suppress_st_warning=True)
def rand_forest_model(xtrain, ytrain, xtest):
    # Import necessary libraries
    # from sklearn.ensemble import RandomForestClassifier

    # Create callable object for RandomForestClassifier
    rand_forest = RandomForestClassifier(bootstrap=True,
                                         class_weight=None,
                                         criterion='gini',
                                         max_depth=None,
                                         max_features='auto',
                                         max_leaf_nodes=None,
                                         min_impurity_decrease=0.0,
                                         min_impurity_split=None,
                                         min_samples_leaf=1,
                                         min_samples_split=2,
                                         min_weight_fraction_leaf=0.0,
                                         n_estimators=10,
                                         n_jobs=None,
                                         oob_score=False,
                                         random_state=69,
                                         verbose=0,
                                         warm_start=False)
    rand_forest.fit(xtrain, ytrain)
    return rand_forest.predict(xtest)


# Predict using the Random Forest Model
yhat_rand_forest = rand_forest_model(xtrain=x_train, ytrain=y_train, xtest=x_test)
# Calculate the K-Nearest Neighbors model accuracy
rand_forest_accuracy = accuracy_score(yhat_rand_forest, y_test)


# Create new dataframe to record all ML's accuracy
model_acc = {'Model': ['Decision Tree',
                       'Logistic Regression',
                       'K-Nearest Neighbors',
                       'Random Forest'],
             'Accuracy_Score': [dec_tree_accuracy,
                                log_reg_accuracy,
                                knn_accuracy,
                                rand_forest_accuracy]}
df_model_acc = pd.DataFrame.from_dict(model_acc)


# Section 5 - Machine Learning Model
def app():
    st.title("Section 5 - Machine Learning Model")
    st.write("""
    In this section, I am going to build multiple Classification Model to predict cuisine based on 
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
    """)  # One-Hot Dataframe already prepared in DataPrep section
    st.write(df_train_onehot.head(3))

    st.write("""---""")

    # DECISION TREE MODEL
    st.subheader("Model 1 - Decision Tree")
    st.code("""
    DecisionTreeClassifier(class_weight=None, 
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
    """)
    st.write("Decision Tree Accuracy - ", dec_tree_accuracy*100, "%")

    st.write("""---""")

    # LOGISTIC REGRESSION MODEL
    st.subheader("Model 2 - Logistic Regression")
    st.code("""
    LogisticRegression(C=1.0,
                       class_weight=None,
                       dual=False,
                       fit_intercept=True,
                       intercept_scaling=1,
                       l1_ratio=None,
                       multi_class='auto',
                       n_jobs=None,
                       penalty='l2',
                       random_state=None,
                       solver='liblinear',
                       tol=0.0001,
                       verbose=0,
                       warm_start=False)
    """)
    st.write("Logistic Regression Accuracy - ", log_reg_accuracy*100, "%")

    st.write("""---""")

    # K-NEAREST NEIGHBORS MODEL
    st.subheader("Model 3 - K-Nearest Neighbors")
    st.code("""
    KNeighborsClassifier(algorithm='auto',
                         leaf_size=30,
                         metric='minkowski',
                         metric_params=None,
                         n_jobs=None,
                         n_neighbors=20,
                         p=2,
                         weights='uniform')
    """)
    st.write("K-Nearest Neighbors Accuracy - ", knn_accuracy*100, "%")

    st.write("""---""")

    # RANDOM FOREST MODEL
    st.subheader("Model 4 - Random Forest")
    st.code("""
    RandomForestClassifier(bootstrap=True,
                           class_weight=None,
                           criterion='gini',
                           max_depth=None,
                           max_features='auto',
                           max_leaf_nodes=None,
                           min_impurity_decrease=0.0,
                           min_impurity_split=None,
                           min_samples_leaf=1,
                           min_samples_split=2,
                           min_weight_fraction_leaf=0.0,
                           n_estimators=10,
                           n_jobs=None,
                           oob_score=False,
                           random_state=None,
                           verbose=0,
                           warm_start=False)
    """)
    st.write("Random Forest Accuracy - ", rand_forest_accuracy*100, "%")

    st.write("""---""")

    st.subheader("Model Accuracy")
    st.write("""
    Out of all the models that were build, there are one particular model that has the best accuracy,
    which is the Logistic Regression model with 78% accuracy.
    """)
    st.dataframe(df_model_acc)
