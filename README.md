
# Graduate Admission Prediction Documentation

This project predicts the chance of admission to graduate programs based on features like GRE Score, TOEFL Score, University Rating, Statement of Purpose (SOP), Letter of Recommendation (LOR), CGPA, and Research experience.

## Dataset Overview
- **Source**: `Admission_Predict_Ver1.1.csv` .
- **Columns**:
  - Serial No.: Unique identifier (not used in modeling).
  - GRE Score: 260–340.
  - TOEFL Score: 0–120.
  - University Rating: 1–5.
  - SOP: Statement of Purpose strength (1–5).
  - LOR: Letter of Recommendation strength (1–5).
  - CGPA: 6–10.
  - Research: Binary (0 or 1).
  - Chance of Admit: Target variable (0–1, probability of admission).

## Workflow
1. **Data Preprocessing**: Load and clean the dataset, handle missing values, and scale features.
2. **Model Training**: Train Linear Regression, Random Forest, and XGBoost models.
3. **Streamlit App**: Deploy a web app for user-friendly predictions using trained models.
4. **Deployment**: Use ngrok to make the Streamlit app publicly accessible.

## Instructions
- Ensure `Admission_Predict_Ver1.1.csv`, `rf_model.pkl`, `xgb_model.pkl`, and `scaler.pkl` are in the working directory.
- Install dependencies: `pip install pandas numpy scikit-learn xgboost tensorflow streamlit pyngrok`.
- Run the Streamlit app: `streamlit run app.py --server.port 8501`.
- Use ngrok for public access: Configure with your ngrok auth token.
