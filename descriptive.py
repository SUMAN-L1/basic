import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats
from sklearn.decomposition import PCA

# Set up the app
st.title("Descriptive Statistics and Advanced Analytics")

# File uploader
uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", type=['csv', 'xlsx'])

# Function to read file
def read_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        return None

# If a file is uploaded
if uploaded_file:
    df = read_file(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())
    
    # Basic Descriptive Statistics
    st.subheader("Basic Descriptive Statistics")
    st.write(df.describe())

    # Correlation Analysis
    st.subheader("Correlation Matrix")
    corr_matrix = df.corr()
    st.write(corr_matrix)
    
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Pairwise Scatter Plots
    st.subheader("Pairwise Scatter Plots")
    pair_plot = sns.pairplot(df)
    st.pyplot(pair_plot)

    # Principal Component Analysis (PCA)
    st.subheader("Principal Component Analysis (PCA)")
    n_components = st.slider("Select number of PCA components", 1, min(len(df.columns), 10), 2)
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df.select_dtypes(include=[np.number]))
    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
    st.write(pca_df)

    # Scree Plot
    st.subheader("Scree Plot")
    explained_variance = pca.explained_variance_ratio_
    fig, ax = plt.subplots()
    ax.plot(range(1, len(explained_variance) + 1), explained_variance, 'o-')
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance")
    st.pyplot(fig)

    # Correlation Circle
    st.subheader("Correlation Circle")
    pca_components = pca.components_
    fig, ax = plt.subplots()
    for i, v in enumerate(df.select_dtypes(include=[np.number]).columns):
        ax.arrow(0, 0, pca_components[0, i], pca_components[1, i], head_width=0.05, head_length=0.05, color='b')
        ax.text(pca_components[0, i]*1.1, pca_components[1, i]*1.1, v, color='r')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Correlation Circle")
    st.pyplot(fig)
    
    # Clustered Heatmap
    st.subheader("Clustered Heatmap")
    fig, ax = plt.subplots()
    sns.clustermap(corr_matrix, annot=True, cmap='coolwarm')
    st.pyplot(fig)
    
    # Displaying Data Types
    st.subheader("Data Types")
    st.write(df.dtypes)

    # Missing Values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Mode
    st.subheader("Mode of Each Column")
    st.write(df.mode().iloc[0])

    # Variance and Standard Deviation
    st.subheader("Variance and Standard Deviation")
    st.write("Variance:\n", df.var())
    st.write("Standard Deviation:\n", df.std())

    # Skewness and Kurtosis
    st.subheader("Skewness and Kurtosis")
    st.write("Skewness:\n", df.skew())
    st.write("Kurtosis:\n", df.kurt())

else:
    st.write("Please upload a file to get started.")

