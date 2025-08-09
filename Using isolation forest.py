import pandas as pd                       
import numpy as np                        # For numerical operations
from sklearn.ensemble import IsolationForest       # Unsupervised anomaly detection model
from sklearn.preprocessing import StandardScaler   # For scaling feature values
from sklearn.metrics import classification_report, confusion_matrix  # For evaluation
import matplotlib.pyplot as plt           # For visualization
import seaborn as sns                     # Better-looking plots

# -----------------------------------------------
# Step 1: Load the dataset
# -----------------------------------------------

# Read the CSV file (downloaded from Kaggle)
df = pd.read_csv('/content/creditcard.csv')

# Show first few rows to understand structure (optional)
print(df.head())

# -----------------------------------------------
# Step 2: Preprocess the data
# -----------------------------------------------

# For faster testing, use only a small random sample (optional)
df = df.sample(frac=0.1, random_state=42)

# The dataset has a 'Class' column:
# 0 = normal transaction, 1 = fraud
# We save this for evaluation later
true_labels = df['Class']

# Drop the 'Class' column so model doesn't see labels (unsupervised learning!)
df = df.drop(['Class'], axis=1)

# Scale the features so that all values are on a similar range
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# -----------------------------------------------
# Step 3: Apply Isolation Forest for anomaly detection
# -----------------------------------------------

# Create the model — set contamination to approx. % of fraud cases
model = IsolationForest(n_estimators=100, contamination=0.001, random_state=42)

# Train the model on the scaled data
model.fit(df_scaled)

# Predict anomalies: 1 = normal, -1 = anomaly (fraud)
predictions = model.predict(df_scaled)

# Convert predictions to match our format:
# 0 = normal, 1 = fraud
predictions = np.where(predictions == 1, 0, 1)

# -----------------------------------------------
# Step 4: Evaluate the model
# -----------------------------------------------

# Show how well the model detected fraud
print("Confusion Matrix:")
print(confusion_matrix(true_labels, predictions))

print("\nClassification Report:")
print(classification_report(true_labels, predictions))

# -----------------------------------------------
# Step 5: Visualize the results
# -----------------------------------------------

# Add the predictions to the original DataFrame
df['Predicted'] = predictions
df['True_Label'] = true_labels.values
df['Amount'] = df['Amount']  # Keep Amount column for plotting
df['Time'] = df['Time']      # Keep Time column for plotting

# Scatter plot: Time vs Amount, color by prediction
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Time', y='Amount', hue='Predicted', data=df, palette='coolwarm')
plt.title('Isolation Forest: Fraud Detection')
plt.xlabel('Time (in seconds)')
plt.ylabel('Transaction Amount')
plt.legend(title='Prediction (0 = Normal, 1 = Fraud)')
plt.grid(True)
plt.show()
