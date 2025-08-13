# Sustainability Predictor

**Problem:** Need a fast, data-driven way to estimate sustainability impact from project features.  
**Solution:** Built a streamlined ML prototype using Python to predict sustainability scores based on input attributes, employing data preprocessing and regression modeling for inference.

## Features
- Accepts structured project data as CSV or JSON, preprocesses features (normalization, encoding).
- Trains a regression model (e.g., Linear Regression or Random Forest) to estimate sustainability scores.
- Outputs prediction with a breakdown of top contributing features.

## Tech Stack
Python, pandas, NumPy (data handling)  
scikit-learn (regression model)  
joblib (model serialization, optional)

## Project Structure
Sustainability_Predictor/  
├── data/ → sample datasets (sample_projects.csv)  
├── src/  
│   ├── preprocess.py → feature cleaning and encoding  
│   ├── model.py → train and evaluate regression model  
│   └── predict.py → inference script for new data  
├── README.md → project overview (this file)  
└── requirements.txt → library dependencies

## Getting Started
1. **Install dependencies**  
   `pip install -r requirements.txt`  
2. **Prepare your data**  
   Place input project features in CSV or JSON format under `data/`.  
3. **Train the model**  
   `python src/model.py --data data/sample_projects.csv --output model.pkl`  
4. **Run predictions**  
   `python src/predict.py --model model.pkl --input data/your_project.csv`  

## Why It Matters
This prototype allows quick estimation of project sustainability scores to help decision-makers compare environmental impacts before committing to project execution.

## Future Enhancements
- Add cross-validation and model performance metrics (R², RMSE)  
- Include SHAP feature importance for interpretability  
- Expand to regression + classification models for risk categorization  
- Build a simple web interface for user-friendly score lookup  

## License & Contributions
Feel free to fork, improve, or extend the model. If you use this for any analysis or project, please credit accordingly.
