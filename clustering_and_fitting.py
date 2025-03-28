
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


def plot_relational_plot(df):
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


def plot_categorical_plot(df):
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


def plot_statistical_plot(df):
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


def statistical_analysis(df, col=None):
    """
    Calculates and prints descriptive statistics including head, describe,
    mean, variance, skewness, and kurtosis of numerical columns.
    Also explains the meaning of each statistical moment.
    """
    numeric_columns = df.select_dtypes(include=[np.number])
    print("\nHead:")
    print(df.head())

    print("\nDescribe:")
    print(numeric_columns.describe())

    print("\nMean (The average value — 1st moment):")
    print(numeric_columns.mean())

    print("\nVariance (How spread out the values are — 2nd moment):")
    print(numeric_columns.var())

    print("\nSkewness (Symmetry of distribution — 3rd moment):")
    print(numeric_columns.skew())

    print("\nKurtosis (Tailedness of distribution — 4th moment):")
    print(numeric_columns.kurtosis())

    return (
        numeric_columns.mean()[col],
        numeric_columns.std()[col],
        numeric_columns.skew()[col],
        numeric_columns.kurtosis()[col]
    )


def preprocessing(df):
    """
    Preprocess the dataset. Placeholder for describe, head, correlation, etc.
    """
    return df


def writing(moments, col):
    """
    Writes out the statistical moments in a readable format.
    """
    print(f"\nFor the attribute {col}:")
    print(f"Mean = {moments[0]:.2f}")
    print(f"Standard Deviation = {moments[1]:.2f}")
    print(f"Skewness = {moments[2]:.2f}")
    print(f"Excess Kurtosis = {moments[3]:.2f}")


def plot_elbow_method():
    fig, ax = plt.subplots()
    plt.savefig("elbow_plot.png")
    return


def one_silhouette_inertia():
    _score = _inertia = None
    return _score, _inertia


def plot_clustered_data(labels, data, xmeans, ymeans, centre_labels):
    """
    Plot clustered data points using the labels returned by KMeans.
    """
    # Recreate the original 2D data for plotting
    X = data  # We expect this to be the scaled data
    if X is None or labels is None:
        print("No data provided for plotting clustered data.")
        return

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    plt.title('Clustered Data Plot')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter, label='Cluster Label')
    plt.savefig("clustering.png")
    plt.show()


def perform_clustering(df, col1, col2):
    """
    Perform KMeans clustering on two selected columns.
    Also saves elbow plot and silhouette plot.
    Returns the cluster labels for plotting.
    """
    from sklearn.metrics import silhouette_samples

    # Step 1: Prepare data
    X = df[[col1, col2]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Elbow Plot
    inertia = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    plt.figure()
    plt.plot(range(1, 10), inertia, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.savefig("elbow_plot.png")
    plt.show()

    # Step 3: Final Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # Step 4: Silhouette Score
    sil_score = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score: {sil_score:.2f}")

    # Step 5: Silhouette Plot
    sample_scores = silhouette_samples(X_scaled, labels)
    plt.figure(figsize=(8, 6))
    y_lower = 10
    for i in range(3):  # for 3 clusters
        ith_scores = sample_scores[labels == i]
        ith_scores.sort()
        size = ith_scores.shape[0]
        plt.barh(range(y_lower, y_lower + size), ith_scores)
        y_lower += size + 10

    plt.axvline(sil_score, color="red", linestyle="--")
    plt.title("Silhouette Scores per Cluster")
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Clustered Samples")
    plt.savefig("silhouette_plot.png")
    plt.show()

    return labels


def perform_fitting(df, col1, col2):
    """
    Fit a Linear Regression model to one feature and one target.
    Returns scaled feature, prediction, and actual target.
    """
    X = df[[col1]].values
    y = df[col2].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    return y_pred, X_scaled, y


def plot_fitted_data(data, X, y):
    """
    Plot actual vs. predicted data with R² score.
    """
    fig, ax = plt.subplots()
    ax.plot(X, y, label='Actual')
    ax.plot(X, data, label='Predicted', linestyle='--')
    ax.set_title(f"Fitting Plot (R²: {r2_score(y, data):.2f})")
    ax.legend()
    plt.savefig("fitting.png")
    return


def main():
    """
    Main function that loads the dataset, cleans it,
    and calls functions to perform analysis, visualization,
    clustering, and regression fitting.
    """
    df = pd.read_csv('data.csv')
    df.rename(columns={'ï»¿Year': 'Year'}, inplace=True)
    df['Year'] = df['Year'].astype(int)
    df = preprocessing(df)

    col = 'Total'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)

    clustering_data = df[['Total', 'Year']]  # The data you used for clustering
    scaler = StandardScaler()
    clustering_data_scaled = scaler.fit_transform(clustering_data)

    clustering_results = perform_clustering(df, 'Total', 'Year')
    plot_clustered_data(
        clustering_results,
        clustering_data_scaled,
        None,
        None,
        None
    )
    fitting_results = perform_fitting(df, 'Year', 'Total')
    plot_fitted_data(*fitting_results)

    return


if __name__ == "__main__":
    main()
