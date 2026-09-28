# US Health Insurance Cost Analysis & Predictive Modelling

An end-to-end data analytics project on what is associated with individual health insurance charges: exploratory analysis, statistical testing, interaction analysis, predictive modelling and an interactive Streamlit dashboard.

**[[Live dashboard →]([YOUR-STREAMLIT-LINK](https://us-health-insurance-cost-analysis-predictive-modelling-cmxk4xg.streamlit.app/))](https://us-health-insurance-cost-analysis-predictive-modelling-cmxk4xg.streamlit.app/)** 
**[Analysis notebook →](notebook/US_health_insurance_analysis.ipynb)**

## At a glance

- **Smokers are charged about 3.8× more** than non-smokers on average ($32,050 vs $8,434, p < 0.001).
- **BMI matters mainly for smokers.** The BMI × smoking interaction is significant (p < 0.001): about +$1,467 per BMI point for smokers vs +$24 for non-smokers.
- **Region is statistically significant but small** (ANOVA p = 0.031, η² = 0.0066, under 1% of the variance).
- **A Random Forest predicts charges well** (test R² = 0.874, MAE $2,549), but some customers are badly under-predicted.

---

# 1. Business Problem

Health insurance charges vary considerably between individuals. Descriptive comparisons alone don't show whether an observed difference is real or just noise, and they don't show whether charges can be estimated from a customer's profile. This project combines statistical testing and predictive modelling to answer:

- Which customer characteristics are associated with insurance charges?
- Do smokers and non-smokers have different average charges?
- Do charges differ across regions, and by how much in practice?
- Does the relationship between BMI and charges differ by smoking status?
- Which variables matter most for predicting charges, and how accurately can charges be predicted?
- Where does the predictive model perform less accurately?

---

# 2. Talk About the Data

The dataset comes from Kaggle and contains **1,338 insured individuals**.

**Source:** [ADD KAGGLE DATASET TITLE AND LINK] (credit to the original author).

| Variable | Description | Type |
|---|---|---|
| `age` | Age of the insured individual | Numerical |
| `sex` | Sex of the individual | Categorical |
| `bmi` | Body Mass Index | Numerical |
| `children` | Number of children/dependants | Numerical |
| `smoker` | Smoking status | Categorical |
| `region` | Residential region | Categorical |
| `charges` | Individual medical insurance charges (USD) | Numerical |

There are no missing values. The target variable for modelling is `charges`, and categorical variables were one-hot encoded before modelling.

The exploratory analysis covered the age, BMI and charge distributions, correlations between numerical variables, and charges by smoking status, region and sex.

---

## Results of the Analysis

### 1. Exploratory Analysis

- The dataset contains 1,338 customer records.
- Age is broadly distributed across the observed range, BMI is centred around approximately 30, and insurance charges are strongly right-skewed.

<img width="1589" height="390" alt="distribution analysis" src="https://github.com/user-attachments/assets/2d9f3610-f92b-4137-a9cb-71d432db9cbe" />


### 2. Smoking Status

Smoking status showed the largest observed difference in insurance charges.

| Group | Average Charges |
|---|---:|
| Non-smoker | $8,434 |
| Smoker | $32,050 |

- Smokers had approximately **3.8× higher average charges** than non-smokers.
- Welch's t-test indicated a statistically significant difference (**p < 0.001**).

<img width="713" height="470" alt="charges by smoking status" src="https://github.com/user-attachments/assets/fbf395d6-d6ad-46b3-83ad-787365fd64bd" />


### 3. Regional & Sex Differences

Insurance charges differed across regions, with the Southeast having the highest average charges and the Southwest the lowest. The difference was statistically significant (p = 0.031), but the effect was small (η² = 0.0066).

Male customers had slightly higher median charges than female customers, although the distributions were similar and overlapped considerably.

<img width="1390" height="490" alt="charges by region   sex" src="https://github.com/user-attachments/assets/dea027fa-bb43-4cfd-a566-b3e8b48859fa" />


### 4. BMI × Smoking Interaction

The association between BMI and insurance charges differed substantially by smoking status.

| Group | Estimated change in charges per BMI point |
|---|---:|
| Non-smokers | **+$23.53** (not significant, p = 0.36) |
| Smokers | **+$1,466.63** |

- The interaction term (about +$1,443 per BMI point) is significant at **p < 0.001**.
- Smokers with BMI ≥ 30 average $41,558, versus $21,363 for smokers below 30.
- For non-smokers the gap is small ($8,843 vs $7,977).
- The straight-line model is therefore an approximation of what looks like a step near BMI 30.

<img width="889" height="590" alt="relationship between bmi   charges based on smoking status" src="https://github.com/user-attachments/assets/138ca0e5-54c7-4e3b-837a-1cf12da81973" />

### 5. Predictive Modelling

Random Forest identified smoking status as the most important feature,
followed by BMI and age.

![Feature Importance](images/random_forest_feature_importance.png)

| Model | Test R² | RMSE | MAE |
|---|---:|---:|---:|
| Linear Regression | 0.784 | $5,796 | $4,181 |
| Random Forest | **0.874** | **$4,421** | **$2,549** |

### 6. Smoking status is the most important predictive feature

A Random Forest (300 trees, max depth 5) was trained on age, BMI, children, sex, smoker and region. Its feature importances:

| Feature | Importance |
|---|---:|
| Smoker | 0.68 |
| BMI | 0.18 |
| Age | 0.12 |
| Children | 0.01 |
| Region | < 0.01 |
| Sex | < 0.01 |

Importance shows how much the fitted model relies on a variable, not that the variable causes higher charges.

<img width="790" height="490" alt="random forest-feature importance" src="https://github.com/user-attachments/assets/ca7da1a6-2548-414b-9d1f-6c2e0d1d9ab5" />

### 7. Prediction Performance

Models were evaluated on an 80/20 hold-out split and with 5-fold cross-validation on all 1,338 rows (no hyperparameter tuning was done).

| Model | Test R² | Test RMSE | Test MAE | 5-fold CV R² |
|---|---:|---:|---:|---:|
| Linear regression | 0.784 | $5,796 | $4,181 | 0.740 ± 0.058 |
| Random Forest | **0.874** | **$4,421** | **$2,549** | **0.852 ± 0.032** |

The Random Forest explains about 87% of the variation in the held-out charges, and its average absolute error is about $2,549. A log-transformed target helped the Random Forest slightly (MAE about $2,088) but made the linear model clearly worse (R² 0.607), so it was not adopted.

<img width="691" height="487" alt="image" src="https://github.com/user-attachments/assets/d9bd5098-a2e0-42c9-a57f-864714e07986" />

The model follows observed charges well for most customers, but 13 of the 214 non-smokers in the test set have actual charges more than **$10,000 above** the prediction. This suggests that important cost drivers, such as medical conditions, are not captured in the dataset.

<img width="700" height="491" alt="image" src="https://github.com/user-attachments/assets/7f89b4a5-e11a-4f9f-ab8a-20953af0b69b" />

## Interactive Streamlit dashboard

The findings are presented as an interactive dashboard, organised as a story: **overview → key findings → BMI × smoking → predictive modelling → interactive prediction.**

- Overview KPIs and charges by smoking status and region (sidebar filters apply here).
- Key-finding statistics computed live from the data.
- The fitted BMI × smoking model over the raw data, plus charges by BMI category.
- Model metrics, actual vs predicted, and feature importance.
- A prediction tool that returns an estimate, a typical range, and a smoker vs non-smoker what-if.

<img width="1917" height="905" alt="image" src="https://github.com/user-attachments/assets/374d201b-3400-4b1c-a200-40b4af7ba265" />

---

# 4. Limitations & Next Steps

## Limitations

- **Limited variables.** Only seven variables. Medical history, conditions, utilisation, medication and plan details are missing, and likely explain part of the largest prediction errors.
- **Observational data.** Results show associations, not causation. Higher charges among smokers are not proof that smoking alone causes them.
- **Unknown provenance.** The source does not explain how the data was collected, so the results may not generalise to other populations, insurers or healthcare systems.
- **Feature importance** is impurity-based and best read as a rough ranking.
- **Individual predictions are uncertain.** The prediction range in the dashboard comes from cross-validated errors for each smoking group. It describes typical error, not the exact uncertainty for one profile.

## Next steps

- Model the BMI ≥ 30 step for smokers explicitly instead of a straight line.
- Try gradient boosting and hyperparameter tuning, and compare alternative importance methods (for example permutation importance).
- Segment prediction errors in more detail.
- Validate on an external dataset with additional healthcare variables.

---

# Project Workflow

```text
Raw Dataset
     ↓
Data Validation & Preparation
     ↓
Exploratory Data Analysis
     ↓
Statistical Hypothesis Testing
     ↓
BMI × Smoking Interaction Analysis
     ↓
Predictive Modelling
     ↓
Model Evaluation & Diagnostics
     ↓
Interactive Streamlit Dashboard
```

# Run It Locally

```bash
git clone YOUR-REPO-URL
cd US-Health-Insurance-Analysis
pip install -r requirements.txt
streamlit run dashboard.py
```

# Tools & Technologies

- **Analysis:** Python, pandas, NumPy, SciPy, statsmodels
- **Machine learning:** scikit-learn (linear regression, Random Forest, cross-validation)
- **Visualisation:** Matplotlib, Seaborn, Plotly
- **Dashboard:** Streamlit
- **Development:** Jupyter Notebook, VS Code, GitHub

# Project Structure

```text
US-Health-Insurance-Analysis/
├── data/
│   └── insurance.csv
├── notebook/
│   └── US_health_insurance_analysis.ipynb
├── images/
│   ├── dashboard_overview.png
│   ├── charges_by_smoking_status.png
│   ├── charges_by_region.png
│   ├── bmi_smoking_relationship.png
│   ├── feature_importance.png
│   ├── actual_vs_predicted.png
│   └── prediction_diagnostics.png
├── dashboard.py
├── requirements.txt
└── README.md
```

# Key Takeaway

Smoking status had the largest observed difference in insurance charges in this dataset, and BMI is associated with higher charges mainly for smokers. Regional differences are statistically significant but small, and largely reflect differences in smoking and BMI. The Random Forest reaches R² = 0.874 on held-out data, while its largest errors show the limits of predicting individual charges from a few demographic and lifestyle variables.

