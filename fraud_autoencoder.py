# Fraud Detection Using PyOD AutoEncoder 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pyod.models.auto_encoder import AutoEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Loading dataset
df = pd.read_csv("creditcard.csv")

print("Dataset loaded successfully!")
print(df.head())
print("Shape:", df.shape)

# Spliting features & labels
X = df.drop("Class", axis=1)
y = df["Class"]   # 0 = normal, 1 = fraud

# Normalizing data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Training/Testing split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Building AutoEncoder model 
model = AutoEncoder(
    hidden_neurons=[32, 16, 8, 16, 32],   # Deep bottleneck structure
    epochs=20,
    batch_size=32,
    contamination=0.001,  # 0.1% fraud rate
    verbose=1
)

# Training model
model.fit(X_train)

# Predictions
y_pred = model.predict(X_test)

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

# Reconstructing error (anomaly scores)
scores = model.decision_scores_

plt.figure(figsize=(10,5))
sns.histplot(scores, bins=50, kde=True)
plt.title("Reconstruction Error Distribution (AutoEncoder)")
plt.xlabel("Reconstruction error")
plt.ylabel("Frequency")
plt.show()

print("\nAnalysis completed successfully!")
