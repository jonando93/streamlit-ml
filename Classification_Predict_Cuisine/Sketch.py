import streamlit as st
from DecisionTree_DataUnderstanding import df_train
import re

# Create variable for Decision Tree called 'mango_tree'
mango_tree = tree.DecisionTreeClassifier(max_depth=5)
mango_tree.fit(ingredients, cuisines)

# Plot the Decision Tree
export_graphviz(mango_tree,
                feature_names=list(ingredients.columns.values),
                out_file='mango_tree.dot',
                class_names=np.unique(cuisines),
                filled=True,
                node_ids=True,
                special_characters=True,
                impurity=False,
                label='all',
                leaves_parallel=False)

with open("mango_tree.dot") as mango_tree_image:
    mango_tree_graph = mango_tree_image.read()
mango_tree_graphviz = graphviz.Source(mango_tree_graph)

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()

with header:
    st.title('asd')
    st.text('asdf')

with dataset:
    st.title('asd')
    st.text('asdf')

with features:
    st.title('asd')
    st.text('asdf')

with model_training:
    st.title('asd')
    st.text('asdf')
