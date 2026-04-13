import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

file_path = "/content/Crop_recommendation.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

X = df.drop(columns=["label"])
y = df["label"]

le = LabelEncoder()
y = le.fit_transform(y)

# Initialize Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42,
    criterion="gini"
)

model.fit(X, y)

print("Enter the required values:")
N = float(input("Nitrogen (N): "))
P = float(input("Phosphorus (P): "))
K = float(input("Potassium (K): "))
temperature = float(input("Temperature (°C): "))
humidity = float(input("Humidity (%): "))
ph = float(input("pH Level: "))
rainfall = float(input("Rainfall (mm): "))

user_input = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=X.columns)

prediction = model.predict(user_input)
predicted_crop = le.inverse_transform(prediction)[0]

print(f"Recommended Crop: {predicted_crop}")
