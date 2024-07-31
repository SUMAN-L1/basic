import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime

# Set up the app
st.title("Descriptive Statistics and Advanced Analytics by SumanEcon")

# File uploader
uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", type=['csv', 'xlsx'])

# Function to read file
def read_file(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            return pd.read_excel(file)
        else:
            st.error("Unsupported file format")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# If a file is uploaded
if uploaded_file:
    df = read_file(uploaded_file)
    if df is not None:
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

        # Feature Importance
        st.subheader("Feature Importance")
        target_column = st.selectbox("Select the target column for feature importance analysis", df.columns)
        if target_column:
            X = df.drop(columns=[target_column])
            y = df[target_column]
            X = pd.get_dummies(X, drop_first=True)  # Handle categorical variables
            if y.nunique() <= 2:  # Binary classification
                model = RandomForestClassifier()
            else:  # Regression
                model = RandomForestRegressor()
            model.fit(X, y)
            feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.write(feature_importance)
            fig, ax = plt.subplots()
            feature_importance.plot(kind='bar', ax=ax)
            st.pyplot(fig)

        # Time Series Analysis
        st.subheader("Time Series Analysis")
        date_column = st.selectbox("Select the date column for time series analysis", df.columns)
        if date_column:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            df.set_index(date_column, inplace=True)
            st.line_chart(df)

            # Trend Analysis
            st.subheader("Trend Analysis")
            fig, ax = plt.subplots()
            for col in df.select_dtypes(include=[np.number]).columns:
                sns.lineplot(data=df[col], label=col, ax=ax)
            st.pyplot(fig)

        # Outlier Detection
        st.subheader("Outlier Detection")
        method = st.selectbox("Select outlier detection method", ["Z-score", "IQR"])
        if method == "Z-score":
            z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
            outliers = (z_scores > 3).sum(axis=1)
            st.write("Number of outliers detected per row:", outliers)
        elif method == "IQR":
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum(axis=1)
            st.write("Number of outliers detected per row:", outliers)

        # Hypothesis Testing
        st.subheader("Hypothesis Testing")
        test_column = st.selectbox("Select the column for hypothesis testing", df.columns)
        if test_column:
            st.write(f"Performing t-test on {test_column}")
            t_stat, p_val = stats.ttest_1samp(df[test_column].dropna(), 0)
            st.write(f"T-statistic: {t_stat}, P-value: {p_val}")

    else:
        st.error("Failed to read the uploaded file. Please check the file format and try again.")
else:
    st.write("Please upload a file to get started.")

# Discussion Section
st.title("Discussion and Decision-Making Guidelines")

# Basic Descriptive Statistics Discussion
st.subheader("Basic Descriptive Statistics")
st.write("""
- **Mean**: Provides the average value of the data. If the mean is significantly higher or lower than expected, it might indicate an issue or an area of interest.
- **Median**: The middle value when data is sorted. If the median differs significantly from the mean, the data might be skewed.
- **Standard Deviation**: Measures the dispersion of data from the mean. A high standard deviation indicates high variability, while a low standard deviation indicates that data points are close to the mean.
- **Variance**: The square of the standard deviation, used to understand data spread.
- **Skewness**: Indicates asymmetry in the data distribution. Positive skewness means a longer tail on the right side, and negative skewness means a longer tail on the left side.
- **Kurtosis**: Indicates the "tailedness" of the data distribution. High kurtosis means heavy tails, while low kurtosis means light tails.
""")

# Correlation Analysis Discussion
st.subheader("Correlation Analysis")
st.write("""
- **Correlation Matrix**: Shows the correlation coefficients between variables. A high positive value indicates a strong positive relationship, while a high negative value indicates a strong negative relationship.
- **Heatmap**: Visual representation of the correlation matrix. Look for strong correlations (both positive and negative) to identify relationships between variables. Be cautious of multicollinearity in predictive models.
""")

# Pairwise Scatter Plots Discussion
st.subheader("Pairwise Scatter Plots")
st.write("""
- **Pairwise Scatter Plots**: Show the relationship between pairs of variables. Look for linear or non-linear relationships and identify potential outliers. Patterns can help in choosing appropriate features for modeling.
""")

# Principal Component Analysis (PCA) Discussion
st.subheader("Principal Component Analysis (PCA)")
st.write("""
- **Explained Variance**: Indicates the amount of variance explained by each principal component. Select the number of components that explain a sufficient amount of variance (e.g., 80%).
- **Scree Plot**: Helps in deciding the number of principal components to retain.
- **Correlation Circle**: Shows the relationship between original variables and principal components. Variables close to each other are positively correlated, while variables opposite to each other are negatively correlated.
""")

# Clustered Heatmap Discussion
st.subheader("Clustered Heatmap")
st.write("""
- **Clustered Heatmap**: Helps in identifying groups of variables that are similar to each other. Use this information to reduce dimensionality or to create features that capture similar information.
""")

# Feature Importance Discussion
st.subheader("Feature Importance")
st.write("""
- **Feature Importance**: Identifies which features are most important for predicting the target variable. Focus on the most important features for building predictive models. Drop less important features to reduce model complexity and improve performance.
""")

# Time Series Analysis Discussion
st.subheader("Time Series Analysis")
st.write("""
- **Line Chart**: Visualize trends over time. Identify any seasonal patterns, trends, or anomalies. Use this information to forecast future values or to understand historical performance.
""")

# Trend Analysis Discussion
st.subheader("Trend Analysis")
st.write("""
- **Trend Analysis**: Visualize long-term trends in the data. Consistent upward or downward trends can indicate growing or declining performance, which is useful for strategic planning and forecasting.
""")

# Outlier Detection Discussion
st.subheader("Outlier Detection")
st.write("""
- **Z-score Method**: Points with a Z-score above 3 or below -3 are considered outliers. Investigate these outliers to understand if they are errors or significant events.
- **IQR Method**: Points outside the range of [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are considered outliers. Similar to the Z-score method, investigate these points to determine their cause.
""")

# Hypothesis Testing Discussion
st.subheader("Hypothesis Testing")
st.write("""
- **T-test**: Compares the mean of a sample to a known value (e.g., 0). A low p-value (typically < 0.05) indicates that the sample mean is significantly different from the known value. Use this information to make data-driven decisions and validate assumptions.
""")
