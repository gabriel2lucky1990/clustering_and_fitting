"""
Olympics Medals Analysis: Visualizations, Statistics, Clustering, and
Regression Fitting
Author: Gabriel Lucky Lotanna (student id: 24070357)
Description: This script explores Olympic data using statistical analysis,
clustering (KMeans), and linear regression.
It includes visualizations (relational, categorical, statistical), statistical
moments, and fitting with proper scaling.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np


def plot_relational(df):
    """
    Generates a relational plot showing total medals by Olympic year.
    Uses a seaborn stripplot to display trends over time.
    """
    if 'Total' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.stripplot(data=df, x='Year', y='Total', jitter=True, size=5)
        plt.title('Olympic Years vs. Total Medals Awarded')
        plt.xlabel('Year')
        plt.ylabel('Total Medals')
        plt.xticks(rotation=45)
        plt.savefig("relational_plot.png")
        plt.show()


def plot_categorical(df):
    """
    Displays a categorical bar plot of the top 10 countries by total medals.
    Groups data by NOC and aggregates total medals.
    """
    if 'NOC' in df.columns and 'Total' in df.columns:
        top_countries = (
            df.groupby('NOC')['Total']
            .sum()
            .sort_values(ascending=False)
            .head(10)
         )

        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_countries.index, y=top_countries.values)
        plt.title('Top 10 Countries by Total Medals (Overall)')
        plt.ylabel('Total Medals')
        plt.xticks(rotation=45)
        plt.savefig("categorical_plot.png")
        plt.show()


def plot_statistical(df):
    """
    Displays a heatmap of correlations between numeric columns in the dataset.
    Useful for identifying relationships between medal counts and other
    variables.
    """
    numeric_columns = df.select_dtypes(include=[np.number])
    if not numeric_columns.empty:
        plt.figure(figsize=(8, 6))
        sns.heatmap(numeric_columns.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Between Medal Counts')
        plt.savefig("statistical_plot.png")
        plt.show()


def calculate_statistics(df):
    """
    Calculates and prints descriptive statistics including head, describe,
    mean, variance, skewness, and kurtosis of numerical columns.
    """
    numeric_columns = df.select_dtypes(include=[np.number])
    print("\nHead:")
    print(df.head())

    print("\nDescribe:")
    print(numeric_columns.describe())

    print("\nMean:")
    print(numeric_columns.mean())

    print("\nVariance:")
    print(numeric_columns.var())

    print("\nSkewness:")
    print(numeric_columns.skew())

    print("\nKurtosis:")
    print(numeric_columns.kurtosis())


def perform_clustering(df):
    """
    Applies KMeans clustering on Total medals and Year columns.
    Displays Elbow and Silhouette plots, and a scatterplot of clustered data.
    Includes scaling and inverse scaling.
    """
    if 'Total' in df.columns and 'Year' in df.columns:
        X_cluster = df[['Total', 'Year']]
        scaler_cluster = StandardScaler()
        X_scaled = scaler_cluster.fit_transform(X_cluster)

        distortions = []
        silhouette_scores = []
        K_range = range(2, 10)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            distortions.append(kmeans.inertia_)
            score = silhouette_score(X_scaled, kmeans.labels_)
            silhouette_scores.append(score)

        plt.figure(figsize=(10, 5))
        plt.plot(K_range, distortions, marker='o')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.savefig("elbow_plot.png")
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(K_range, silhouette_scores, marker='o')
        plt.title('Silhouette Score vs. Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.savefig("silhouette_plot.png")
        plt.show()

        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df,
            x='Year',
            y='Total',
            hue='Cluster',
            palette='Set2'
        )

        plt.title('Grouping Olympic Results into Clusters')
        plt.savefig("clustering_plot.png")
        plt.show()


def perform_fitting(df):
    """
    Fits a linear regression model using Year as the independent variable
    and Total medals as the dependent variable. Shows actual vs. predicted
    plots and residuals. Includes feature scaling and inverse transformation.
    """
    if {'Year', 'Total'}.issubset(df.columns):
        X_fit = df[['Year']]
        y_fit = df[['Total']]

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled_fit = scaler_X.fit_transform(X_fit)
        y_scaled_fit = scaler_y.fit_transform(y_fit)

        model = LinearRegression()
        model.fit(X_scaled_fit, y_scaled_fit)

        predicted_scaled = model.predict(X_scaled_fit)
        predicted = scaler_y.inverse_transform(predicted_scaled)
        residuals = y_fit.values.flatten() - predicted.flatten()

        print("\nRegression coefficient for Year:", model.coef_[0][0])
        print("Intercept:", model.intercept_[0])
        print("R² score:", r2_score(y_fit, predicted))

        plt.figure(figsize=(8, 6))
        plt.scatter(y_fit, predicted)
        plt.xlabel('Actual Total Medals')
        plt.ylabel('Predicted Total Medals')
        plt.title('Actual vs. Predicted Medals')
        plt.savefig("fitting_plot.png")
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.scatter(predicted, residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Total Medals')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.savefig("residual_plot.png")
        plt.show()


def main():
    """
    Main function that loads the dataset, cleans it,
    and calls functions to perform analysis, visualization,
    clustering, and regression fitting.
    """
    df = pd.read_csv('data.csv')
    df.rename(columns={'ï»¿Year': 'Year'}, inplace=True)
    df['Year'] = df['Year'].astype(int)

    print(df.head())
    print("Unique years:", df['Year'].unique())

    plot_relational(df)
    plot_categorical(df)
    plot_statistical(df)
    calculate_statistics(df)
    perform_clustering(df)
    perform_fitting(df)


if __name__ == "__main__":
    main()
