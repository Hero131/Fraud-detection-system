import pandas as pd                       
import numpy as np                         
from sklearn.ensemble import IsolationForest       
from sklearn.preprocessing import StandardScaler   
from sklearn.metrics import classification_report, confusion_matrix  
import matplotlib.pyplot as plt          
import seaborn as sns                     

df = pd.read_csv('/content/creditcard.csv')
print(df.head())

df = df.sample(frac=0.1, random_state=42)
true_labels = df['Class']
df = df.drop(['Class'], axis=1)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

model = IsolationForest(n_estimators=100, contamination=0.001, random_state=42)
model.fit(df_scaled)
predictions = model.predict(df_scaled)
predictions = np.where(predictions == 1, 0, 1)

print("Confusion Matrix:")
print(confusion_matrix(true_labels, predictions))

print("\nClassification Report:")
print(classification_report(true_labels, predictions))

df['Predicted'] = predictions
df['True_Label'] = true_labels.values
df['Amount'] = df['Amount']  
df['Time'] = df['Time']     

plt.figure(figsize=(12, 6))
sns.scatterplot(x='Time', y='Amount', hue='Predicted', data=df, palette='coolwarm')
plt.title('Isolation Forest: Fraud Detection')
plt.xlabel('Time (in seconds)')
plt.ylabel('Transaction Amount')
plt.legend(title='Prediction (0 = Normal, 1 = Fraud)')
plt.grid(True)
plt.show()
