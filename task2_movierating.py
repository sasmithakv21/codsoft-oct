import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("datasets/IMDb Movies India.csv", encoding="latin1")

print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())
# Convert columns to numeric values

df["Year"] = df["Year"].str.extract(r"(\d{4})")[0].astype(float)

df["Duration"] = df["Duration"].str.extract(r"(\d+)")[0].astype(float)

df["Votes"] = df["Votes"].str.replace(",", "", regex=False)
df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")

df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

print("\nAfter converting numeric columns:")
print(df[["Year", "Duration", "Votes", "Rating"]].head())

print("\nMissing values in important columns:")
print(df[["Year", "Duration", "Votes", "Rating"]].isnull().sum())
# Remove rows where Rating is missing
df = df.dropna(subset=["Rating"])

# Fill missing values in input columns with their median
df["Year"] = df["Year"].fillna(df["Year"].median())
df["Duration"] = df["Duration"].fillna(df["Duration"].median())
df["Votes"] = df["Votes"].fillna(df["Votes"].median())

print("\nMissing values after cleaning:")
print(df[["Year", "Duration", "Votes", "Rating"]].isnull().sum())
# Select features and target

X = df[["Year", "Duration", "Votes"]]
y = df["Rating"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)
from sklearn.linear_model import LinearRegression

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")
# Predict ratings for test data
y_pred = model.predict(X_test)

print("\nPredicted Ratings:")
print(y_pred[:10])
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)
# Predict rating for a new movie

new_movie = pd.DataFrame({
    "Year": [2024],
    "Duration": [150],
    "Votes": [5000]
})

predicted_rating = model.predict(new_movie)

print("\nNew Movie Details:")
print(new_movie)

print("\nPredicted Movie Rating:", predicted_rating[0])
df.to_csv("cleaned_movie_data.csv", index=False)

print("\nCleaned dataset saved successfully!")
import os

print(os.path.exists("cleaned_movie_data.csv"))
plt.figure(figsize=(8, 5))

plt.hist(df["Rating"], bins=20)

plt.xlabel("Movie Rating")
plt.ylabel("Number of Movies")
plt.title("Distribution of Movie Ratings")

plt.show()
plt.figure(figsize=(8, 5))

plt.scatter(df["Votes"], df["Rating"], alpha=0.5)

plt.xlabel("Number of Votes")
plt.ylabel("Movie Rating")
plt.title("Votes vs Movie Rating")

plt.show()
plt.figure(figsize=(8, 5))

plt.scatter(df["Votes"], df["Rating"], alpha=0.5)

plt.xlabel("Number of Votes")
plt.ylabel("Movie Rating")
plt.title("Votes vs Movie Rating")

plt.show()
