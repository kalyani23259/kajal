from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model & encoders
model = pickle.load(open("house_model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()

    # Encode categorical
    for col in label_encoders:
        if col in data:
            data[col] = label_encoders[col].transform([data[col]])[0]

    # Convert all to float
    features = [float(v) for v in data.values()]
    final_features = [np.array(features)]

    prediction = model.predict(final_features)[0]
    prediction = round(prediction, 2)

    return render_template("index.html", prediction_text=f"🏡 Estimated House Price: ₹{prediction:,.2f}")

if __name__ == "__main__":
    app.run(debug=True)
