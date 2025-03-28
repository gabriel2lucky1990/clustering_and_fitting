"""
Olympics Medals Analysis: Visualizations, Statistics, Clustering, and
Regression Fitting
Author: Gabriel Lucky Lotanna
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

# Load the Olympics dataset
df = pd.read_csv('data.csv')

# Display first rows to understand structure
print(df.head())
print(df.columns)

# Clean column names and rename Year column
df.rename(columns={'ï»¿Year': 'Year'}, inplace=True)
df['Year'] = df['Year'].astype(int)

# Check for unique years to confirm dataset range
print("Unique years:", df['Year'].unique())

# Step 1: Relational Plot
if 'Total' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.stripplot(data=df, x='Year', y='Total', jitter=True, size=5)
    plt.title('Olympic Years vs. Total Medals Awarded')
    plt.xlabel('Year')
    plt.ylabel('Total Medals')
    plt.xticks(rotation=45)
    plt.show()

# Step 2: Categorical Plot - Top Countries by Total Medals
if 'NOC' in df.columns and 'Total' in df.columns:
    top_countries = (
        df.groupby('NOC')['Total'].sum()
        .sort_values(ascending=False).head(10)
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_countries.index, y=top_countries.values)
    plt.title('Top 10 Countries by Total Medals (Overall)')
    plt.ylabel('Total Medals')
    plt.xticks(rotation=45)
    plt.show()

# Step 3: Statistical Plot - Correlation Heatmap
numeric_columns = df.select_dtypes(include=[np.number])
if not numeric_columns.empty:
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_columns.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Between Medal Counts')
    plt.show()

# Step 4: Statistical Moments
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

# Step 5: KMeans Clustering with Elbow and Silhouette Methods
if 'Total' in numeric_columns.columns and 'Year' in numeric_columns.columns:
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
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

    plt.figure(figsize=(10, 5))
    plt.plot(K_range, distortions, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(K_range, silhouette_scores, marker='o')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x='Year',
        y='Total',
        hue='Cluster',
        palette='Set2')
    plt.title('Grouping Olympic Results into Clusters')
    plt.show()

# Step 6: Regression Fitting with Diagnostics and Scaling
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
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(predicted, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Total Medals')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()
