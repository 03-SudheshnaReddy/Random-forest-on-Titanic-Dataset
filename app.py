import joblib
import pandas as pd
import gradio as gr

# 1. Load model + feature names
model = joblib.load("titanic_rf.pkl")
feature_names = joblib.load("titanic_features.pkl")

# 2. Define a prediction function
def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    # manual one-hot encoding to match training
    data = {
        "pclass": [pclass],
        "age": [age],
        "sibsp": [sibsp],
        "parch": [parch],
        "fare": [fare],
        "sex_male": [1 if sex == "male" else 0],
        "embarked_Q": [1 if embarked == "Q" else 0],
        "embarked_S": [1 if embarked == "S" else 0],
    }

    df = pd.DataFrame(data)

    # ensure all expected columns exist, even if not used
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]

    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0, 1]  # probability of survival

    label = "Survived" if pred == 1 else "Did NOT survive"
    return f"{label} (probability of survival = {proba:.2f})"

# 3. Build the Gradio interface
inputs = [
    gr.Slider(1, 3, step=1, label="Passenger Class (1 = 1st, 3 = 3rd)"),
    gr.Radio(["male", "female"], label="Sex"),
    gr.Slider(0, 80, step=1, label="Age"),
    gr.Slider(0, 8, step=1, label="Number of Siblings/Spouses (sibsp)"),
    gr.Slider(0, 6, step=1, label="Number of Parents/Children (parch)"),
    gr.Slider(0, 600, step=10, label="Fare"),
    gr.Radio(["C", "Q", "S"], label="Port of Embarkation"),
]

output = gr.Textbox(label="Prediction")

demo = gr.Interface(
    fn=predict_survival,
    inputs=inputs,
    outputs=output,
    title="Titanic Survival Predictor",
    description="Enter passenger details to predict if they would survive the Titanic disaster."
)

if __name__ == "__main__":
    demo.launch()
