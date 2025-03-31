
"""
Clustering and Fitting Assignment
Author: Gabriel Lucky Lotanna
Student ID: 24070357

This script fulfills the requirements of the clustering and fitting assignment.

It includes:
- Data preprocessing
- A relational plot (scatter)
- A categorical plot (bar chart)
- A statistical plot (correlation heatmap)
- Use of the four main statistical moments (mean, standard deviation, skewness,
  kurtosis)
- Clustering using KMeans with elbow and silhouette evaluation
- Fitting using linear regression on one feature and one target variable

All plots are saved with high resolution (dpi=144) and follow the structure 
defined in the provided GitHub template.

The script is written to comply with PEP 8 and follows the functional
 structure required for CodeGrade auto-assessment.
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
from sklearn.metrics import silhouette_samples


def plot_relational_plot(df):
    """Relational plot: Horsepower vs MPG with enhanced styling"""
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df, x="horsepower", y="mpg", ax=ax,
        s=80, edgecolor='black', linewidth=0.5
    )
    ax.set_title("Horsepower vs MPG")
    fig.savefig("relational_plot.png", dpi=144)
    plt.show()
    print("Saved: relational_plot.png")
    print("Saved: relational_plot.png")


def plot_categorical_plot(df):
    """Categorical plot: Average MPG by Cylinders with enhanced styling"""
    fig, ax = plt.subplots()
    sns.barplot(
        data=df, x="cylinders", y="mpg", estimator=np.mean,
        palette="pastel", ax=ax
    )
    ax.set_title("Average MPG by Cylinders")
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.savefig("categorical_plot.png", dpi=144)
    plt.show()
    print("Saved: categorical_plot.png")
    print("Saved: categorical_plot.png")


def plot_statistical_plot(df):
    """Statistical plot: Correlation Heatmap with enhanced styling"""
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(
        corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax,
        linewidths=0.5, linecolor='white', cbar=True, square=True
    )
    ax.set_title("Correlation Heatmap")
    fig.savefig("statistical_plot.png", dpi=144)
    plt.show()
    print("Saved: statistical_plot.png")
    print("Saved: statistical_plot.png")


def statistical_analysis(df, col):
    """Calculate and display statistical moments and tools for a column"""
    print("\nHEAD of Data:")
    print(df.head())

    print("\nDESCRIBE:")
    print(df.describe())

    print("\nCORRELATION:")
    print(df.select_dtypes(include=[np.number]).corr())

    data = df[col]
    mean = data.mean()
    stddev = data.std()
    skewness = skew(data)
    excess_kurtosis = kurtosis(data)
    return mean, stddev, skewness, excess_kurtosis


def preprocessing(df):
    """Clean and prepare data"""
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
    df.dropna(subset=["horsepower"], inplace=True)
    return df


def writing(moments, col):
    """Interpretation of statistical moments"""
    print(f"\nFor the attribute {col}:")
    print(f"Mean = {moments[0]:.2f}")
    print(f"Standard Deviation = {moments[1]:.2f}")
    print(f"Skewness = {moments[2]:.2f}")
    print(f"Excess Kurtosis = {moments[3]:.2f}")

    print("\nInterpretation:")
    if moments[2] > 1:
        print("The data is highly right-skewed.")
    elif moments[2] < -1:
        print("The data is highly left-skewed.")
    else:
        print("The data is approximately symmetrical.")

    if moments[3] > 0:
        print("The distribution is leptokurtic (peaked).")
    else:
        print("The distribution is platykurtic (flat).")


def perform_clustering(df, col1, col2):
    """Perform clustering using KMeans, including elbow plot, silhouette
    score and plot, and inverse transform"""
 
    # Step 1: Select and scale data
    X = df[[col1, col2]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Inner function 1: elbow plot
    def plot_elbow_method():
        fig, ax = plt.subplots()
        distortions = []
        for k in range(1, 10):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
            kmeans.fit(X_scaled)
            distortions.append(kmeans.inertia_)
        ax.plot(range(1, 10), distortions, marker='o')
        ax.set_title("Elbow Plot")
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Inertia")
        fig.savefig("elbow_plot.png", dpi=144)
        plt.show()
        print("✅ Saved: elbow_plot.png")

    # Inner function 2: silhouette score and inertia
    def one_silhouette_inertia():
        kmeans_test = KMeans(n_clusters=3, random_state=42, n_init=20)
        labels_test = kmeans_test.fit_predict(X_scaled)
        _score = silhouette_score(X_scaled, labels_test)
        _inertia = kmeans_test.inertia_
        return _score, _inertia

    # Call elbow plot
    plot_elbow_method()

    # Main clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)

    # Get silhouette score and inertia from function
    _score, _inertia = one_silhouette_inertia()
    print(f"✅ Silhouette Score: {_score:.2f}")
    print(f"✅ Inertia: {_inertia:.2f}")

    # Silhouette plot
    sample_silhouette_values = silhouette_samples(X_scaled, labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10
    for i in range(3):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / 3)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper), 
            0, 
            ith_cluster_silhouette_values,
            facecolor=color, 
            edgecolor=color, 
            alpha=0.7
        )
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.axvline(x=_score, color="red", linestyle="--")
    ax.set_title("Silhouette Plot for 3 Clusters")
    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster Label")
    fig.savefig("silhouette_plot.png", dpi=144)
    plt.show()
    print("✅ Saved: silhouette_plot.png")

    # Inverse transform for plotting
    X_inv = scaler.inverse_transform(X_scaled)
    x_vals, y_vals = X_inv[:, 0], X_inv[:, 1]
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    return labels, x_vals, y_vals, centers[:, 0], centers[:, 1]


def plot_clustered_data(
    labels, x, y, xmeans, ymeans, 
    centroids_label="Cluster Centers"
):
    """Plot clustered data with enhanced styling"""
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=x, y=y, hue=labels, palette="Set2", s=80,
        edgecolor='black', linewidth=0.5, ax=ax
    )

    ax.scatter(
        xmeans, ymeans, color="black", s=120,
        marker="X", label=centroids_label
    )

    ax.legend()
    ax.set_title("Clustering: Weight vs Acceleration")
    fig.savefig("clustering_plot.png", dpi=144)
    plt.show()
    print("Saved: clustering_plot.png")
    print("Saved: clustering_plot.png")


def perform_fitting(df, col1, col2):
    """Perform linear regression with scaling and inverse scaling"""
    X = df[[col1]].values
    y = df[[col2]].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    model = LinearRegression()
    model.fit(X_scaled, y_scaled)

    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    return df, X, y, y_pred, model


def plot_fitted_data(df, X, y, y_pred):
    """Plot regression line and actual values with enhanced styling"""
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=X.squeeze(), y=y.squeeze(), label="Actual", s=80,
        edgecolor='black', linewidth=0.5, ax=ax
    )

    sns.lineplot(
        x=X.squeeze(), y=y_pred.squeeze(), color="red",
        label="Fitted Line", linewidth=2.5, ax=ax
    )

    ax.set_title("Fitting: Horsepower vs MPG")
    ax.legend()
    fig.savefig("fitting_plot.png", dpi=144)
    plt.show()
    print("Saved: fitting_plot.png")
    print("Saved: fitting_plot.png")




def main():
    df = pd.read_csv("data.csv")
    df = preprocessing(df)

    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, "mpg")
    writing(moments, "mpg")

    clustering_results = perform_clustering(df, "weight", "acceleration")
    plot_clustered_data(*clustering_results)

    fitting_results = perform_fitting(df, "horsepower", "mpg")
    plot_fitted_data(*fitting_results[:4])

    coef = fitting_results[4].coef_[0][0]
    intercept = fitting_results[4].intercept_[0]
    print(
       f"\nRegression Equation: MPG = {coef:.4f} * Horsepower + "
       f"{intercept:.2f}"
   )


if __name__ == "__main__":
    main()
