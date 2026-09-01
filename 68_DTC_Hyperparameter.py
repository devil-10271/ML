import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Page setup
st.set_page_config(page_title="Decision Tree Classifier", layout="wide")

st.sidebar.title("Decision Tree Classifier")

# Sidebar Form with Calculate Button
with st.sidebar.form("hyperparameters_form"):
    criterion = st.selectbox("Criterion", ["gini", "entropy", "log_loss"])
    splitter = st.selectbox("Splitter", ["best", "random"])

    # Max Depth
    max_depth_input = st.number_input("Max Depth (0 for None)", min_value=0, max_value=50, value=0, step=1)
    max_depth = None if max_depth_input == 0 else int(max_depth_input)

    # Min Samples Split & Leaf
    min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=375, value=2)
    min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=375, value=1)
    
    # Max Features
    max_features = st.slider("Max Features", min_value=1, max_value=2, value=2)

    # Max Leaf Nodes
    max_leaf_nodes_input = st.number_input("Max Leaf Nodes (0 for None)", min_value=0, max_value=100, value=0, step=1)
    max_leaf_nodes = None if max_leaf_nodes_input == 0 else int(max_leaf_nodes_input)

    # Min Impurity Decrease
    min_impurity_decrease = st.slider(
        "Min Impurity Decrease", 
        min_value=0.0, 
        max_value=0.5, 
        value=0.0, 
        step=0.01,
        format="%.2f"
    )

    # Calculate Button
    submitted = st.form_submit_button("Calculate")

# Generate Dataset
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)

# Train Model
clf = DecisionTreeClassifier(
    criterion=criterion,
    splitter=splitter,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    max_features=max_features,
    max_leaf_nodes=max_leaf_nodes,
    min_impurity_decrease=min_impurity_decrease,
    random_state=42
)
clf.fit(X, y)

# Accuracy Calculation
y_pred = clf.predict(X)
acc = accuracy_score(y, y_pred)

# Display Accuracy Score
st.metric(label="Model Accuracy", value=f"{acc * 100:.2f}%")

# Decision Boundary Plot (Reduced Size)
fig_boundary, ax_boundary = plt.subplots(figsize=(5, 3.5))

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax_boundary.contourf(xx, yy, Z, alpha=0.1, cmap='rainbow')
ax_boundary.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow', edgecolors='none', s=25)
ax_boundary.set_xlim(x_min, x_max)
ax_boundary.set_ylim(y_min, y_max)
ax_boundary.grid(True, linestyle='--', alpha=0.5)

# Render compact plot container
col1, _ = st.columns([1, 1])
with col1:
    st.pyplot(fig_boundary, use_container_width=False)

# Decision Tree Diagram below graph
st.markdown("**Decision Tree Structure**")
fig_tree, ax_tree = plt.subplots(figsize=(12, 5))
plot_tree(
    clf,
    feature_names=["Feature 1", "Feature 2"],
    class_names=["Class 0", "Class 1"],
    filled=True,
    rounded=True,
    fontsize=7,
    ax=ax_tree
)
st.pyplot(fig_tree)