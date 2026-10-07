import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("datasets/IRIS.csv")

print("First 5 rows:")
print(df.head())


# --------------------------------------------------
# 2. DATASET INFORMATION
# --------------------------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())


# --------------------------------------------------
# 3. CHECK SPECIES
# --------------------------------------------------

print("\nUnique Species:")
print(df["species"].unique())

print("\nSpecies Count:")
print(df["species"].value_counts())


# --------------------------------------------------
# 4. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

X = df[
    ["sepal_length", "sepal_width",
     "petal_length", "petal_width"]
]

y = df["species"]


# --------------------------------------------------
# 5. SPLIT DATA INTO TRAINING AND TESTING
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# --------------------------------------------------
# 6. CREATE AND TRAIN MODEL
# --------------------------------------------------

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

print("\nModel training completed successfully!")


# --------------------------------------------------
# 7. MAKE PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(X_test)

print("\nPredicted Values:")
print(y_pred)

print("\nActual Values:")
print(y_test.values)


# --------------------------------------------------
# 8. CALCULATE ACCURACY
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

print("\nModel Accuracy Percentage:")
print(accuracy * 100, "%")


# --------------------------------------------------
# 9. CLASSIFICATION REPORT
# --------------------------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------------------------
# 10. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------------------------
# 11. CONFUSION MATRIX VISUALIZATION
# --------------------------------------------------

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Iris Species - Confusion Matrix")

plt.show()


# --------------------------------------------------
# 12. VISUALIZE SPECIES DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="species"
)

plt.title("Iris Species Distribution")
plt.xlabel("Species")
plt.ylabel("Count")

plt.show()


# --------------------------------------------------
# 13. TEST WITH A NEW FLOWER
# --------------------------------------------------

new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("\nNew Flower Prediction:")
print(prediction[0])