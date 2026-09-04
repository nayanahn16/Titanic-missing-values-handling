# Titanic Missing Values Handling and Exploratory Data Analysis

**Academic Minor Project**  
**Student Level:** 3rd-Year B.E. Artificial Intelligence and Data Science  

---

## 1. Introduction

The **Titanic Dataset** is an iconic benchmark dataset in Data Science and Machine Learning. It records demographic profiles, ticket info, travel classes, family relations, and survival outcomes of passengers aboard the RMS Titanic, which sank on April 15, 1912.

In real-world data science projects, raw data is rarely perfect. Datasets frequently contain missing entries, noise, or improper formats. This project focuses on **Exploratory Data Analysis (EDA)**, **Missing Value Detection**, and **Missing Value Handling/Imputation** using Python, Pandas, Seaborn, Matplotlib, and Scikit-Learn.

---

## 2. Problem Statement

Real-world datasets regularly suffer from missing or incomplete records caused by data collection gaps, non-responses, or system failures. In the Titanic dataset, attributes such as passenger age, cabin location (`deck`), and port of embarkation contain missing values. 

The primary problem addressed in this project is:
- How to systematically identify missing values and quantify missingness percentages.
- How to determine the appropriate imputation approach (median vs. mode vs. column removal) without biasing the dataset.
- How to perform statistical data exploration and build an interactive web dashboard for viva demonstrations.

---

## 3. Objectives

- Explore the structural characteristics and data types of the Titanic dataset.
- Detect missing values and compute column-wise missing percentages.
- Visualize missing data patterns using heatmaps.
- Analyze passenger demographics (Gender, Age, Passenger Class).
- Perform Survival Analysis across demographics.
- Analyze Ticket Fare distributions.
- Conduct Family Relationship analysis and derive `family_size`.
- Impute missing numerical data (`age`) using **Median Imputation**.
- Impute missing categorical data (`embarked`, `embark_town`) using **Mode Imputation**.
- Demonstrate standardized imputation using Scikit-Learn's `SimpleImputer`.
- Justify and remove columns with excessive missingness (`deck`).
- Compare missing data counts before vs. after cleaning.
- Generate and export `titanic_cleaned.csv`.
- Develop an interactive **Streamlit Dashboard** (`app.py`).

---

## 4. Technologies Used

- **Programming Language:** Python 3.x
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning Imputation:** Scikit-Learn (`SimpleImputer`)
- **Dashboard Framework:** Streamlit
- **Interactive Environment:** Jupyter Notebook

---

## 5. Dataset Description

The Titanic dataset contains 891 rows and 15 attributes:

| Column Name | Data Type | Description | Missing Values (Raw) |
|---|---|---|---|
| `survived` | Integer | 0 = Did not survive, 1 = Survived | 0 |
| `pclass` | Integer | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) | 0 |
| `sex` | Object | Passenger gender (male / female) | 0 |
| `age` | Float | Passenger age in years | 177 (~19.87%) |
| `sibsp` | Integer | Number of siblings / spouses aboard | 0 |
| `parch` | Integer | Number of parents / children aboard | 0 |
| `fare` | Float | Ticket fare paid ($) | 0 |
| `embarked` | Object | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) | 2 (~0.22%) |
| `class` | Object | Text representation of class | 0 |
| `who` | Object | Man, woman, or child | 0 |
| `adult_male` | Boolean | True if adult male | 0 |
| `deck` | Object | Cabin deck level (A–G) | 688 (~77.22%) |
| `embark_town` | Object | Full port town name | 2 (~0.22%) |
| `alive` | Object | 'yes' or 'no' | 0 |
| `alone` | Boolean | True if traveling alone | 0 |

---

## 6. Project Workflow

```
Raw Dataset (data/titanic.csv)
       ↓
Data Exploration & Summary Statistics (head, tail, describe, info)
       ↓
Missing Value Detection & Percentage Table (isnull().sum())
       ↓
Missing Value Visualization (Seaborn Heatmap)
       ↓
Exploratory Data Analysis (Gender, Age, Class, Survival, Fare, Family Size)
       ↓
Missing Value Handling & Imputation:
  - Age → Median Imputation (28.0 years)
  - Embarked / Embark Town → Mode Imputation ('S' / 'Southampton')
  - Deck → Column Drop (>75% missingness)
  - SimpleImputer Demonstration
       ↓
Before vs After Cleaning Comparison & Verification
       ↓
Correlation Heatmap Analysis
       ↓
Export Cleaned Dataset (titanic_cleaned.csv)
       ↓
Interactive Streamlit Web Dashboard (app.py)
```

---

## 7. Missing Value Handling Techniques

1. **Median Imputation (`age`):**
   - **Rationale:** Age contains outliers (infants to 80-year-olds) and is skewed. The mean age (29.7) is sensitive to extreme values, whereas the median age (28.0) provides a robust central estimate.
   - `df['age'].fillna(df['age'].median())`

2. **Mode / Most Frequent Imputation (`embarked`, `embark_town`):**
   - **Rationale:** Categorical data cannot be averaged. Southampton ('S') is the most frequent embarkation port (>70% of passengers).
   - `df['embarked'].fillna(df['embarked'].mode()[0])`

3. **Scikit-Learn `SimpleImputer`:**
   - `SimpleImputer(strategy='median')` for numerical data.
   - `SimpleImputer(strategy='most_frequent')` for categorical data.

4. **Handling Columns with Excessive Missingness (`deck`):**
   - **Rationale:** `deck` has 688 missing entries out of 891 (77.22%). Imputing over 75% of a column distorts dataset integrity. Dropping the column preserves dataset quality.

---

## 8. Exploratory Data Analysis (Key Insights)

- **Gender Survival Rate:** Females had a **~74.2%** survival rate compared to **~18.9%** for males ("Women and children first" policy).
- **Class Survival Rate:** 1st Class passengers had a **~63.0%** survival rate, 2nd Class had **~47.3%**, and 3rd Class had **~24.2%**.
- **Family Size Impact:** `family_size = sibsp + parch + 1`. Solo travelers and large families (>4 members) suffered lower survival rates than small families (size 2–4).

---

## 9. Visualizations

The project contains 10+ core Seaborn/Matplotlib charts:
1. Missing Values Heatmap (Raw Data)
2. Gender Distribution Countplot
3. Age Distribution Histogram with KDE
4. Passenger Class Distribution Plot
5. Overall Survival Distribution Plot
6. Gender vs. Survival Bar Chart
7. Passenger Class vs. Survival Bar Chart
8. Ticket Fare Distribution Histogram
9. Family Size Distribution Plot
10. Missing Values Before vs. After Cleaning Bar Chart
11. Correlation Heatmap (Cleaned Data)

---

## 10. Project Structure

```
Titanic_Missing_Values_Project/
│
├── data/
│   └── titanic.csv                       # Raw Titanic Dataset
│
├── Titanic_Missing_Values_Handling.ipynb # Main Academic Jupyter Notebook (34 Sections)
├── app.py                                # Streamlit Web Dashboard Application
├── requirements.txt                      # Python Library Dependencies
├── README.md                             # Academic Documentation & Viva Preparation Guide
└── titanic_cleaned.csv                   # Exported Cleaned Dataset
```

---

## 11. Installation Instructions (Windows)

Open Command Prompt / PowerShell in your project directory:

```bash
# 1. Create a virtual environment (optional but recommended)
python -m venv venv

# 2. Activate virtual environment on Windows
venv\Scripts\activate

# 3. Install required libraries
pip install -r requirements.txt
```

---

## 12. Run Jupyter Notebook

```bash
jupyter notebook
```
Open **`Titanic_Missing_Values_Handling.ipynb`** in your browser and execute cells sequentially.

---

## 13. Run Streamlit Dashboard

```bash
streamlit run app.py
```
The interactive web app will open automatically at `http://localhost:8501`.

---

## 14. Actual Dataset Results Summary

- **Total Passengers:** 891
- **Total Features:** 15
- **Initial Missing Values:** 867 (Age: 177, Embarked: 2, Embark Town: 2, Deck: 688)
- **Median Age Used for Imputation:** 28.0 years
- **Mode Embarked Used for Imputation:** 'S' (Southampton)
- **Remaining Missing Values after Cleaning:** 0 (100% clean)
- **Cleaned Dataset Dimensions:** 891 rows × 15 columns (with `family_size`, without `deck`)

---

## 15. Conclusion

This project successfully demonstrates the fundamentals of data cleaning, missing value detection, statistical analysis, and interactive dashboard creation for academic submission. By filling missing numerical values with median, mode-imputing categorical variables, dropping heavily incomplete columns, and conducting EDA, we produced a clean dataset `titanic_cleaned.csv` ready for data science applications.

---

## 16. Viva Questions and Answers (Preparation Guide)

### Q1: What is the Titanic dataset?
**Ans:** It is a historic dataset containing passenger demographics, travel classes, fares, family relations, and survival outcomes from the 1912 Titanic disaster.

### Q2: Why is data cleaning necessary before analysis?
**Ans:** Real data contains missing entries, anomalies, and noise. Cleaning ensures statistics are accurate, unskewed, and reliable.

### Q3: What is the difference between Mean, Median, and Mode Imputation?
**Ans:**
- **Mean Imputation:** Fills missing values with the arithmetic average. Sensitive to outliers.
- **Median Imputation:** Fills missing values with the middle value of a sorted dataset. Robust against outliers (ideal for `age`).
- **Mode Imputation:** Fills missing values with the most frequent category (ideal for categorical data like `embarked`).

### Q4: Why did we use Median Imputation for Age?
**Ans:** Age contains extreme values (infants < 1 year to adults 80 years old). Median (28.0) is not affected by extreme values, whereas mean (29.7) can be pulled by outliers.

### Q5: Why did we drop the `deck` column instead of imputing it?
**Ans:** `deck` has 688 missing rows out of 891 (~77% missingness). Imputing over 75% of a column invents synthetic data and distorts analysis.

### Q6: What is Scikit-Learn `SimpleImputer`?
**Ans:** It is a standardized transformer class in `sklearn.impute` that automates missing data replacement using strategies like `'mean'`, `'median'`, and `'most_frequent'`.

### Q7: How is `family_size` calculated?
**Ans:** `family_size = sibsp + parch + 1`, where `sibsp` is siblings/spouses, `parch` is parents/children, and `1` represents the passenger themselves.

### Q8: What insights were observed from the visualizations?
**Ans:** 
1. Females had a significantly higher survival rate (~74%) than males (~19%).
2. 1st Class passengers had the highest survival rate (~63%), while 3rd Class had the lowest (~24%).
3. Small families (size 2–4) survived better than solo travelers or very large families.

---

## 17. Future Enhancements

- Apply advanced imputation algorithms like **KNNImputer** or **MICE (Multivariate Imputation by Chained Equations)**.
- Build predictive Machine Learning models (Logistic Regression, Decision Trees, Random Forest) to predict passenger survival.
- Deploy the Streamlit application to cloud platforms like Streamlit Community Cloud or Render.
