# 🛳️ Titanic Survival Prediction

A machine learning project that predicts whether a passenger survived the Titanic disaster.  
The final model is deployed as an interactive **Gradio web app** on **HuggingFace Spaces**.

---

## 📘 Project Summary

- Cleaned and preprocessed the Titanic dataset  
- Selected key features (Pclass, Sex, Age, Fare, etc.)  
- Trained a **Random Forest Classifier**  
- Achieved ~80% accuracy  
- Saved model using `joblib`  
- Built a simple UI using **Gradio**  
- Deployed publicly on **HuggingFace Spaces**

---

## 📂 Files Included

- `DM_FINAL_LAB.ipynb` – Notebook (EDA + training)  
- `app.py` – Gradio web app  
- `titanic_rf.pkl` – Trained model  
- `titanic_features.pkl` – Feature order for predictions  
- `requirements.txt` – Dependencies  

---

## ▶️ Run Locally

```
pip install -r requirements.txt
python app.py
```
---
## 🛠️ Tech Used
- Python
- Pandas
- Scikit-Learn
- Gradio
- HuggingFace Spaces
