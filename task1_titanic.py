import pandas as pd

# Load dataset
df = pd.read_csv("datasets/Titanic-Dataset.csv")

# Display first 5 rows
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())
# Select useful features
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

X = df[features]
y = df["Survived"]

# Fill missing Age values
X["Age"] = X["Age"].fillna(X["Age"].median())

# Fill missing Embarked values
X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

print("\nMissing values after cleaning:")
print(X.isnull().sum())
# Convert categorical columns into numbers
X = pd.get_dummies(X, columns=["Sex", "Embarked"], drop_first=True)

print("\nFeatures after encoding:")
print(X.head())

print("\nData types:")
print(X.dtypes)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)
from sklearn.linear_model import LogisticRegression

# Create the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")
# Make predictions on test data
y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred[:10])
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
import matplotlib.pyplot as plt
import seaborn as sns

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Survived", "Survived"],
    yticklabels=["Not Survived", "Survived"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Titanic Survival - Confusion Matrix")
plt.show()