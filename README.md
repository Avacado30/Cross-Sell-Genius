
# 🧠 Cross-Sell Genius

**Cross-Sell Genius** is a machine learning-powered recommendation system designed to optimize cross-selling strategies in the retail sector. Leveraging customer demographics and transaction behavior, it predicts the likelihood of purchasing additional products, helping businesses increase customer lifetime value (CLV) and drive targeted marketing campaigns.

---

## 🚀 Features

- 📊 **Data Preprocessing:** Cleans and transforms customer demographic and transactional data.
- 🤖 **Machine Learning Models:** Implements various classification models (Random Forest, XGBoost, Logistic Regression) to predict product affinity.
- 📈 **Model Evaluation:** Uses metrics like F1-score, accuracy, and ROC-AUC to evaluate model performance.
- 🧩 **Explainability:** Incorporates SHAP (SHapley Additive exPlanations) for interpreting model predictions.
- 📎 **Modular Codebase:** Clean and modular implementation for easy customization and extension.

---

## 📁 Repository Structure

```
Cross-Sell-Genius/
│
├── data/                  # Raw and processed datasets (ignored in .gitignore)
├── models/                # Trained model artifacts
├── notebooks/             # Jupyter notebooks for EDA and experimentation
├── src/                   # Source code for training and evaluation
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── shap_analysis.py
├── utils/                 # Helper functions
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── main.py                # Main script to run end-to-end pipeline
```

---

## 🛠️ Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Avacado30/Cross-Sell-Genius.git
   cd Cross-Sell-Genius
   ```

2. **Create Virtual Environment (optional)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💡 Usage

1. **Prepare Data**  
   Place your dataset in the `data/` directory. Modify paths in `main.py` or the relevant scripts as needed.

2. **Run the Full Pipeline**
   ```bash
   python main.py
   ```

   This runs data preprocessing, model training, evaluation, and SHAP-based explainability.

3. **Experiment in Notebooks**  
   Navigate to the `notebooks/` directory and explore:
   - `EDA.ipynb` for exploratory data analysis.
   - `Modeling.ipynb` for model comparisons.
   - `SHAP_Analysis.ipynb` for model interpretation.

---

## 📊 Model Performance

| Model               | Accuracy | F1 Score | ROC-AUC |
|---------------------|----------|----------|---------|
| Random Forest       | 0.89     | 0.87     | 0.91    |
| XGBoost             | 0.90     | 0.88     | 0.92    |
| Logistic Regression | 0.84     | 0.82     | 0.86    |

*Note: Results may vary based on dataset and tuning.*

---

## 🧠 Explainability with SHAP

![SHAP Summary](https://raw.githubusercontent.com/Avacado30/Cross-Sell-Genius/main/assets/shap_summary.png)

SHAP values provide intuitive insights into feature importance, enabling stakeholders to understand and trust the model predictions.

---

## 📌 Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- shap
- joblib
- tqdm

Install them using:

```bash
pip install -r requirements.txt
```

---

## 📚 Future Enhancements

- 🏗️ Integrate with Streamlit for real-time user input and predictions.
- 💾 Connect with SQL/NoSQL databases for dynamic data ingestion.
- 🔍 Incorporate unsupervised learning for customer segmentation.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or feature suggestions
- Fork the repo and create a pull request with improvements

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Aishwarya (Avacado30)**  
📫 [LinkedIn](https://www.linkedin.com/in/your-profile)  
💼 Passionate about ML, data-driven systems, and customer behavior modeling.
