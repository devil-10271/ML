import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree

# Page Configuration
st.set_page_config(
    page_title="Decision Tree Regressor", layout="wide"
)

# ---------------------------------------------------------
# 1. Synthetic Dataset Generation
# ---------------------------------------------------------
np.random.seed(42)
X = np.sort(np.random.uniform(-5, 5, 150)).reshape(-1, 1)
# Generate non-linear continuous data with noise
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# ---------------------------------------------------------
# 2. Sidebar Hyperparameters
# ---------------------------------------------------------
st.sidebar.title("Decision Tree Regressor")

criterion = st.sidebar.selectbox(
    "Criterion",
    ["squared_error", "absolute_error", "friedman_mse", "poisson"],
)
splitter = st.sidebar.selectbox("Splitter", ["best", "random"])

max_depth_input = st.sidebar.number_input(
    "Max Depth (0 for None)", min_value=0, max_value=50, value=0, step=1
)
max_depth = None if max_depth_input == 0 else max_depth_input

min_samples_split = st.sidebar.slider(
    "Min Samples Split", min_value=2, max_value=150, value=2
)
min_samples_leaf = st.sidebar.slider(
    "Min Samples Leaf", min_value=1, max_value=150, value=2
)

max_leaf_nodes_input = st.sidebar.number_input(
    "Max Leaf Nodes (0 for None)", min_value=0, max_value=100, value=5, step=1
)
max_leaf_nodes = None if max_leaf_nodes_input == 0 else max_leaf_nodes_input

min_impurity_decrease = st.sidebar.number_input(
    "Min Impurity Decrease",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.01,
)

# ---------------------------------------------------------
# 3. Model Training & Prediction
# ---------------------------------------------------------
model = DecisionTreeRegressor(
    criterion=criterion,
    splitter=splitter,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    max_leaf_nodes=max_leaf_nodes,
    min_impurity_decrease=min_impurity_decrease,
    random_state=42,
)

model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

# Grid points for smooth step curve
X_test = np.linspace(-5, 5, 500).reshape(-1, 1)
y_test_pred = model.predict(X_test)

# ---------------------------------------------------------
# 4. Main Panel Outputs
# ---------------------------------------------------------

# Display R2 Score above the graph
st.metric(label="R² Score", value=f"{r2:.4f}")

# Main Scatter + Prediction Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(X, y, color="y", edgecolor="k", label="Data Points")
ax.plot(X_test, y_test_pred, color="b", linewidth=2, label="Prediction")
ax.set_xlabel("Feature (Col1)")
ax.set_ylabel("Target")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)

# Tree Structure Visualization
st.subheader("Decision Tree Structure")
fig_tree, ax_tree = plt.subplots(figsize=(12, 6))
plot_tree(
    model,
    feature_names=["Col1"],
    filled=True,
    ax=ax_tree,
    fontsize=9,
    rounded=True,
)
st.pyplot(fig_tree)