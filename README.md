# Customer Churn Prediction Project

## 📌 Overview
This project predicts whether a customer will churn using machine learning techniques. The model is trained on customer data and deployed using a Flask API for real-time predictions.

## 🚀 Features
- Data cleaning and preprocessing
- Handling missing values
- Feature engineering (encoding)
- Random Forest model
- Hyperparameter tuning
- Threshold tuning to improve recall
- Feature importance analysis
- Flask API deployment
- Tested using Postman
- 
## 🧠 Model Performance
- Accuracy: ~75%
- Recall (Churn): ~87%
- Focused on improving recall to detect more churn customers

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Flask
- Joblib
- Postman

## 📂 Project Structure
churn-project/
│── main.py
│── app.py
│── model.pkl
│── columns.pkl


---

## ▶️ How to Run

### 1. Install dependencies
pip install pandas numpy scikit-learn flask joblib

### 2. Train model
python main.py

### 3. Run Flask API
python app.py

## 🧪 API Testing
http://127.0.0.1:5000/predict

POST request:

Example JSON:
{
  "tenure": 12,
  "MonthlyCharges": 70,
  "TotalCharges": 800,
  "Contract_Two year": 1
}

## 💡 Key Insights

- Customers with lower tenure are more likely to churn, indicating that new customers are at higher risk of leaving.
- Customers with higher monthly charges show a higher probability of churn.
- Customers with long-term contracts (two-year plans) are less likely to churn compared to short-term users.
- Total charges are correlated with tenure, where long-term customers tend to stay and contribute more revenue.
- Improving recall helped in identifying more potential churn customers, which is important for business retention strategies.

## 📈 Future Improvements

- Improve model performance using advanced algorithms like XGBoost or Gradient Boosting.
- Perform hyperparameter tuning using GridSearchCV for better optimization.
- Implement cross-validation to improve model reliability.
- Build a user-friendly interface using Streamlit for easier interaction.
- Deploy the application on cloud platforms like AWS or Render.
- Use more real-world datasets to improve generalization.
- Add real-time data input and monitoring for production-level usage.

  ## 👨‍💻 Author
**Sane Hemanth Reddy**

- GitHub: https://github.com/HemanthReddySane
- LinkedIn: (https://www.linkedin.com/in/sane-hemanth-reddy-118971374?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
