
"""
Clustering and Fitting Assignment Template
Student: [Your Name]
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis


def plot_relational_plot(df):
    """Relational plot: Horsepower vs MPG"""
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="horsepower", y="mpg", ax=ax)
    ax.set_title("Horsepower vs MPG")
    plt.savefig("relational_plot.png", dpi=144)
    plt.show()


def plot_categorical_plot(df):
    """Categorical plot: Average MPG by Cylinders"""
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="cylinders", y="mpg", estimator=np.mean, ax=ax)
    ax.set_title("Average MPG by Cylinders")
    plt.savefig("categorical_plot.png", dpi=144)
    plt.show()


def plot_statistical_plot(df):
    """Statistical plot: Correlation Heatmap"""
    fig, ax = plt.subplots()
    corr = df.drop(columns=["car name"]).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.savefig("statistical_plot.png", dpi=144)
    plt.show()


def statistical_analysis(df, col):
    """Calculate and display statistical moments and tools for a column"""
    print("\nHEAD of Data:")
    print(df.head())

    print("\nDESCRIBE:")
    print(df.describe())

    print("\nCORRELATION:")
    print(df.corr())

    data = df[col]
    mean = data.mean()
    stddev = data.std()
    skewness = skew(data)
    excess_kurtosis = kurtosis(data)
    return mean, stddev, skewness, excess_kurtosis


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


def plot_elbow(X_scaled):
    """Elbow plot for KMeans"""
    distortions = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        distortions.append(kmeans.inertia_)

    plt.figure()
    plt.plot(range(1, 10), distortions, marker='o')
    plt.title("Elbow Plot")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.savefig("elbow_plot.png", dpi=144)
    plt.show()


def perform_clustering(df, col1, col2):
    """Perform clustering with scaling, elbow, silhouette, and inverse transform"""
    X = df[[col1, col2]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow plot
    plot_elbow(X_scaled)

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)

    print(f"Silhouette Score: {score:.2f}")

    # Inverse scaling for plotting
    X_inv = scaler.inverse_transform(X_scaled)
    x_vals, y_vals = X_inv[:, 0], X_inv[:, 1]
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    return labels, x_vals, y_vals, centers[:, 0], centers[:, 1]


def plot_clustered_data(labels, x, y, xmeans, ymeans, centroids_label="Cluster Centers"):
    """Plot clustered data with centers"""
    fig, ax = plt.subplots()
    sns.scatterplot(x=x, y=y, hue=labels, palette="Set2", ax=ax)
    plt.scatter(xmeans, ymeans, color="black", s=100, marker="X", label=centroids_label)
    plt.legend()
    plt.title("Clustering: Weight vs Acceleration")
    plt.savefig("clustering_plot.png", dpi=144)
    plt.show()


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
    """Plot regression line and actual values"""
    fig, ax = plt.subplots()
    sns.scatterplot(x=X.squeeze(), y=y.squeeze(), label="Actual", ax=ax)
    sns.lineplot(x=X.squeeze(), y=y_pred.squeeze(), color="red", label="Fitted Line", ax=ax)
    ax.set_title("Fitting: Horsepower vs MPG")
    plt.legend()
    plt.savefig("fitting_plot.png", dpi=144)
    plt.show()


def preprocessing(df):
    """Clean and prepare data"""
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
    df.dropna(subset=["horsepower"], inplace=True)
    return df


def main():
    df = pd.read_csv("Cleaned_auto_mpg.csv")
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
    print(f"\nRegression Equation: MPG = {coef:.4f} * Horsepower + {intercept:.2f}")


if __name__ == "__main__":
    main()
