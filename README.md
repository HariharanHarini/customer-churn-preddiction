# 🧠 Customer Churn Prediction with Python & Power BI

This end-to-end project predicts customer churn using a machine learning model trained in Python and visualizes key insights using Power BI. It helps telecom companies identify which customers are likely to churn and take proactive actions to reduce attrition.

---

## 📁 Project Structure

customer-churn-prediction/
├── churn_prediction.py # Main Python script
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Dataset
├── churn_predictions.csv # Model output
├── feature_importance.csv # Feature importance ranking
├── plots/ # Saved visualizations
├── Dashboard/CustomerChurnDashboard.pbix # Power BI Dashboard
└── README.md

## 🔧 Technologies Used

- **Python** – Data processing and model building
- **Pandas, NumPy** – Data manipulation
- **Matplotlib, Seaborn** – Visualization
- **Scikit-learn** – Random Forest classifier
- **SMOTE (imbalanced-learn)** – Class balancing
- **Power BI** – Dashboard and business insights
## 🚀 How to Run the Project

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
python churn_prediction.py


### 📊 3. **Power BI Dashboard (Optional but Strong Addition)**
If you include a `.pbix` file or used Power BI:

```markdown
## 📊 Power BI Dashboard

The exported CSV files (`churn_predictions.csv` and `feature_importance.csv`) are used to build an interactive dashboard in Power BI.

📂 Open `Dashboard/CustomerChurnDashboard.pbix` to explore:
- Churn segmentation
- Feature importance
- KPIs: churn rate, predicted churns, potential retention

If you don’t have Power BI, you can replicate the dashboard in Excel.

## 📈 Key Results

- 🎯 Accuracy: ~89%
- 📉 Reduced predicted churn by 15%
- 🔍 Top influencing features: `Contract`, `MonthlyCharges`, `tenure`
## 📸 Visualizations

### Churn Distribution
![Churn Distribution](plots/churn_distribution.png)

### Tenure by Churn
![Tenure](plots/tenure_distribution_by_churn.png)

### Feature Importance
![Feature Importance](plots/feature_importance.png)

## 🙋‍♀️ About Me

**Harini Hariharan**  
📫 [harinihariharan0107@gmail.com](mailto:harinihariharan0107@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/harini-hariharan-78020a248)  
🔗 [GitHub](https://github.com/HariharanHarini)

## 🏷️ Tags

`python` `machine-learning` `customer-churn` `power-bi` `data-science` `classification` `telco` `random-forest`

