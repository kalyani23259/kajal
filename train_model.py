import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_excel("house_d1.xlsx")

# Encode categorical columns
categorical_cols = ['PROPERTYTYPE', 'LOCATION', 'FURNISHINGSTATUS', 'HOTWATERHEATING', 'AIRCONDITIONING']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and Target
X = df.drop("PRICE", axis=1)
y = df["PRICE"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model + encoders
pickle.dump(model, open("house_model.pkl", "wb"))
pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))
