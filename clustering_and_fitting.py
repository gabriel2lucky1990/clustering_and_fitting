import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, r2_score
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
# Total vs Year (show all years by adding jitter for better clarity)
if 'Total' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.stripplot(data=df, x='Year', y='Total', jitter=True, size=5)
    plt.title('Olympic Years vs. Total Medals Awarded')
    plt.xlabel('Year')
    plt.ylabel('Total Medals')
    plt.xticks(rotation=45)
    plt.show()

# Explanation
# This plot helps me see how medal distributions
# change across different Olympic years.
# The data points show fluctuations that could be linked
# to varying participation levels and global events.

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

# Explanation: The bar plot highlights which countries have consistently
# performed well
# at the Olympics.
# # The top positions confirm expected results, but it was insightful to see
# the relative dominance.

# Step 3: Statistical Plot - Correlation Heatmap
numeric_columns = df.select_dtypes(include=[np.number])
if not numeric_columns.empty:
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_columns.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Between Medal Counts')
    plt.show()

# Explanation: The heatmap confirms that gold, silver, and bronze
# counts strongly correlate with total medals,
# reinforcing that high total medals come from consistently strong performances

# Step 4: Statistical Moments
print("\nMean:")
print(numeric_columns.mean())

print("\nVariance:")
print(numeric_columns.var())

print("\nSkewness:")
print(numeric_columns.skew())

print("\nKurtosis:")
print(numeric_columns.kurtosis())

# Detailed Discussion of the Four Statistical Moments:
# 1. Mean: The average values show the central tendency of
# the medal counts across different years and countries.
# 2. Variance: This indicates how spread out the medal counts are.
# A high variance shows large differences between countries or years.
# 3. Skewness: Positive skewness in medal totals indicates
# that while most countries win a moderate number of medals,
# a few countries have extremely high totals.
# 4. Kurtosis: High kurtosis suggests the presence of outliers —
# a few instances where medal counts are much higher than the average.
# Overall, these moments give a statistical profile of Olympic performance
# and highlight disparity between average participants and leading countries.
# Step 5: KMeans Clustering with Elbow Method

if 'Total' in numeric_columns.columns and 'Year' in numeric_columns.columns:
    X_cluster = df[['Total', 'Year']]

    distortions = []
    silhouette_scores = []
    K_range = range(2, 10)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_cluster)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_cluster, kmeans.labels_))

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
    df['Cluster'] = kmeans.fit_predict(X_cluster)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x='Year',
        y='Total',
        hue='Cluster',
        palette='Set2')
    plt.title('Grouping Olympic Results into Clusters')
    plt.show()

# Explanation: Clustering helped me visualize groupings in the data.
# I could clearly see how different eras or performance levels cluster together

# Step 6: Regression Fitting with Diagnostics
if {'Year', 'Total'}.issubset(df.columns):
    X_fit = df[['Year']]
    y_fit = df['Total']

    model = LinearRegression()
    model.fit(X_fit, y_fit)

    predicted = model.predict(X_fit)
    residuals = y_fit - predicted

    print("\nRegression coefficient for Year:", model.coef_[0])
    print("Intercept:", model.intercept_)
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

# Explanation: The regression results suggest a gentle trend in medal totals
# over time.
# The prediction is not perfect, but it shows that Olympic performance
# evolves alongside historical and global changes.
# Conclusion: Working on this analysis has helped me understand how to
# clean data, visualize trends, apply clustering, and fit predictive models.
# The Olympic dataset was interesting to explore,
# revealing patterns and dynamics I hadn’t expected.

# Conclusion:
# The analysis conducted demonstrated strong relationships between medal counts
# and Olympic years, with clear clusters emerging after testing multiple
# cluster counts using the elbow and silhouette methods.
# The chosen 3-cluster model effectively grouped historical data based
# on performance trends and Olympic periods.
# Linear regression fitting showed a gradual upward trend in total medals
# awarded, confirmed by the R² value.
# Residual analysis indicated an acceptable model fit, though improvements
# with more complex models could be considered.
# Overall, this project successfully combined data cleaning, visualization,
# clustering, and regression analysis for insightful findings.
