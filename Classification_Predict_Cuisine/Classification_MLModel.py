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


# ----- TEST DATASET -----
# Since the TEST DATASET does not include cuisine column, I will use TRAIN DATASET instead.
ingredients_test = df_test_onehot.iloc[:, 1:]

# ----- TRAIN DATASET -----
# I will split the Train dataset to create 4 variables (xy train & xy test)

# Create 2 new variables for cuisine and ingredients (x & y)
cuisines = df_train_onehot['cuisine'] # y
ingredients = df_train_onehot.iloc[:, 2:] # x

x_train, x_test, y_train, y_test = train_test_split(ingredients, cuisines, test_size=0.2, random_state=69)

# Create object for decision tree model called dec_tree
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
dec_tree.fit(x_train, y_train)
y_pred_dec_tree = dec_tree.predict(x_test)
dec_tree_accuracy = accuracy_score(y_pred_dec_tree, y_test)

# Create object for log reg model called log_reg
log_reg = LogisticRegression(C=1.0,
                             class_weight=None,
                             dual=False,
                             fit_intercept=True,
                             intercept_scaling=1,
                             l1_ratio=None,
                             max_iter=100,
                             multi_class='auto',
                             n_jobs=None,
                             penalty='l2',
                             random_state=69,
                             solver='lbfgs',
                             tol=0.0001,
                             verbose=0,
                             warm_start=False)
log_reg.fit(x_train, y_train)
y_pred_log_reg = log_reg.predict(x_test)
log_reg_accuracy = accuracy_score(y_pred_log_reg, y_test)

# Create object for KNN model called knn
# Add elbow method (NEXT TASK!) to find most suitable n_neighbors
knn = KNeighborsClassifier(algorithm='auto',
                           leaf_size=30,
                           metric='minkowski',
                           metric_params=None,
                           n_jobs=None,
                           n_neighbors=20,
                           p=2,
                           weights='uniform')
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
knn_accuracy = accuracy_score(y_pred_knn, y_test)

# Create object for Random Forest model called rand_forest
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
rand_forest.fit(x_train, y_train)
y_pred_rand_forest = rand_forest.predict(x_test)
rand_forest_accuracy = accuracy_score(y_pred_rand_forest, y_test)

# Create new dataframe to record all ML's accuracy
model_acc = {'Model' : ['Decision Tree',
                        'Logistic Regression',
                        'K-Nearest Neighbors',
                        'Random Forest'],
             'Accuracy_Score' : [dec_tree_accuracy,
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
    """) # One-Hot Dataframe already prepared in DataPrep section
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
                       max_iter=100,
                       multi_class='auto',
                       n_jobs=None,
                       penalty='l2',
                       random_state=None,
                       solver='lbfgs',
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