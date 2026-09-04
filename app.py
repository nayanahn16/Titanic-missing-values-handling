import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page Configuration
st.set_page_config(
    page_title="Titanic Missing Values & EDA Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        font-weight: 400;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Main Title & Header
st.markdown('<div class="main-title">🚢 Titanic Missing Values Handling & EDA Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Academic Minor Project | B.E. Artificial Intelligence and Data Science</div>', unsafe_allow_html=True)

# Data Loading Function with Cache and Fallback
@st.cache_data
def load_datasets():
    raw_path = 'data/titanic.csv'
    clean_path = 'titanic_cleaned.csv'
    
    # Load Raw Dataset
    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
    else:
        url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
        raw_df = pd.read_csv(url)
        os.makedirs('data', exist_ok=True)
        raw_df.to_csv(raw_path, index=False)
        
    # Load Clean Dataset (or compute dynamically if missing)
    if os.path.exists(clean_path):
        clean_df = pd.read_csv(clean_path)
    else:
        clean_df = raw_df.copy()
        clean_df['age'] = clean_df['age'].fillna(clean_df['age'].median())
        if 'embarked' in clean_df.columns:
            clean_df['embarked'] = clean_df['embarked'].fillna(clean_df['embarked'].mode()[0])
        if 'embark_town' in clean_df.columns:
            clean_df['embark_town'] = clean_df['embark_town'].fillna(clean_df['embark_town'].mode()[0])
        clean_df['family_size'] = clean_df['sibsp'] + clean_df['parch'] + 1
        if 'deck' in clean_df.columns:
            clean_df = clean_df.drop(columns=['deck'])
        clean_df.to_csv(clean_path, index=False)
        
    return raw_df, clean_df

# Error Handling block
try:
    df_raw, df_clean = load_datasets()
except Exception as e:
    st.error(f"❌ Error loading Titanic datasets: {e}")
    st.info("Please make sure 'data/titanic.csv' is present in the workspace.")
    st.stop()

# Ensure family_size exists in df_raw for consistent filter display
if 'family_size' not in df_raw.columns:
    df_raw['family_size'] = df_raw['sibsp'] + df_raw['parch'] + 1

# Sidebar Filters
st.sidebar.header("🔍 Interactive Dashboard Filters")

# Gender Filter
gender_options = ["All"] + sorted(list(df_raw['sex'].unique()))
selected_gender = st.sidebar.selectbox("Filter by Gender:", gender_options)

# Class Filter
class_options = ["All"] + sorted(list(df_raw['pclass'].unique()))
selected_class = st.sidebar.selectbox("Filter by Passenger Class:", class_options)

# Survival Filter
survival_options = ["All", "Survived (1)", "Did Not Survive (0)"]
selected_survival = st.sidebar.selectbox("Filter by Survival Status:", survival_options)

# Apply Filters to DataFrames
filtered_raw = df_raw.copy()
filtered_clean = df_clean.copy()

if selected_gender != "All":
    filtered_raw = filtered_raw[filtered_raw['sex'] == selected_gender]
    filtered_clean = filtered_clean[filtered_clean['sex'] == selected_gender]

if selected_class != "All":
    filtered_raw = filtered_raw[filtered_raw['pclass'] == selected_class]
    filtered_clean = filtered_clean[filtered_clean['pclass'] == selected_class]

if selected_survival != "All":
    surv_val = 1 if "Survived (1)" in selected_survival else 0
    filtered_raw = filtered_raw[filtered_raw['survived'] == surv_val]
    filtered_clean = filtered_clean[filtered_clean['survived'] == surv_val]

# Summary KPI Cards
st.subheader("📊 Key Dataset Metrics")
m1, m2, m3, m4 = st.columns(4)

total_passengers = len(filtered_raw)
total_missing_raw = filtered_raw.isnull().sum().sum()
survived_count = (filtered_raw['survived'] == 1).sum() if 'survived' in filtered_raw.columns else 0
survival_rate = (survived_count / total_passengers * 100) if total_passengers > 0 else 0.0

m1.metric("Total Passengers", f"{total_passengers}")
m2.metric("Total Missing Values (Raw)", f"{total_missing_raw}")
m3.metric("Survivors", f"{survived_count}")
m4.metric("Survival Rate", f"{survival_rate:.1f}%")

st.markdown("---")

# Tab Layout
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Dataset Overview", 
    "❓ Missing Values Analysis", 
    "👥 Demographics & Survival", 
    "💰 Fare & Family Analysis", 
    "🧹 Data Cleaning (Before/After)", 
    "🔥 Correlation Heatmap"
])

# TAB 1: Dataset Overview
with tab1:
    st.subheader("Dataset Overview & Exploration")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Raw Dataset Preview (First 10 Rows):**")
        st.dataframe(filtered_raw.head(10), use_container_width=True)
    with col2:
        st.write("**Cleaned Dataset Preview (First 10 Rows):**")
        st.dataframe(filtered_clean.head(10), use_container_width=True)
        
    st.subheader("Numerical Summary Statistics")
    st.dataframe(filtered_raw.describe().round(2), use_container_width=True)

# TAB 2: Missing Values Analysis
with tab2:
    st.subheader("Missing Value Detection & Heatmap")
    
    m_count = filtered_raw.isnull().sum()
    m_pct = (m_count / len(filtered_raw) * 100).round(2) if len(filtered_raw) > 0 else 0
    missing_tbl = pd.DataFrame({'Missing Count': m_count, 'Missing Percentage (%)': m_pct}).sort_values(by='Missing Count', ascending=False)
    
    col_a, col_b = st.columns([1, 1.5])
    with col_a:
        st.write("**Missing Values Summary Table:**")
        st.dataframe(missing_tbl, use_container_width=True)
        
    with col_b:
        st.write("**Missing Values Heatmap (Raw Data):**")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.heatmap(filtered_raw.isnull(), cbar=False, cmap='viridis', yticklabels=False, ax=ax)
        ax.set_title("Missing Values Heatmap (Yellow = Missing)", fontsize=12, fontweight='bold')
        st.pyplot(fig)

# TAB 3: Demographics & Survival
with tab3:
    st.subheader("Passenger Demographics & Survival Analysis")
    
    d1, d2, d3 = st.columns(3)
    with d1:
        st.write("**Gender Distribution**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=filtered_raw, x='sex', palette='pastel', ax=ax)
        ax.set_title("Passenger Gender Distribution")
        st.pyplot(fig)
        
    with d2:
        st.write("**Age Distribution**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(filtered_raw['age'].dropna(), kde=True, bins=25, color='teal', ax=ax)
        ax.set_title("Age Distribution")
        st.pyplot(fig)
        
    with d3:
        st.write("**Passenger Class Distribution**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=filtered_raw, x='pclass', palette='Set2', ax=ax)
        ax.set_title("Passenger Class Distribution")
        st.pyplot(fig)
        
    st.markdown("---")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.write("**Overall Survival Distribution**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=filtered_raw, x='survived', palette='Set1', ax=ax)
        ax.set_title("Survival Distribution (0=No, 1=Yes)")
        st.pyplot(fig)
        
    with s2:
        st.write("**Gender vs Survival**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=filtered_raw, x='sex', hue='survived', palette='muted', ax=ax)
        ax.set_title("Gender vs Survival")
        st.pyplot(fig)
        
    with s3:
        st.write("**Class vs Survival**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=filtered_raw, x='pclass', hue='survived', palette='Accent', ax=ax)
        ax.set_title("Class vs Survival")
        st.pyplot(fig)

# TAB 4: Fare & Family Analysis
with tab4:
    st.subheader("Ticket Fare & Family Relationship Analysis")
    
    f1, f2 = st.columns(2)
    with f1:
        st.write("**Ticket Fare Distribution**")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.histplot(filtered_raw['fare'], bins=30, kde=True, color='purple', ax=ax)
        ax.set_title("Ticket Fare Distribution ($)")
        st.pyplot(fig)
        
    with f2:
        st.write("**Family Size Distribution (SibSp + Parch + 1)**")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.countplot(data=filtered_clean, x='family_size', palette='cool', ax=ax)
        ax.set_title("Family Size Distribution")
        st.pyplot(fig)

# TAB 5: Data Cleaning (Before / After)
with tab5:
    st.subheader("Data Cleaning & Imputation Demonstration")
    
    b_series = df_raw.isnull().sum()
    a_series = df_clean.isnull().sum()
    comp_df = pd.DataFrame({'Missing Before Cleaning': b_series, 'Missing After Cleaning': a_series}).fillna(0)
    
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.write("**Missing Count Comparison Table:**")
        st.dataframe(comp_df, use_container_width=True)
        st.success("✅ 'Age' imputed using Median (28.0)\n✅ 'Embarked' imputed using Mode ('S')\n✅ 'Deck' dropped due to >75% missingness")
        
    with c2:
        st.write("**Before vs After Cleaning Visual Comparison:**")
        cols_present = [c for c in ['age', 'embarked', 'embark_town', 'deck'] if c in df_raw.columns]
        chart_data = pd.DataFrame({
            'Column': cols_present,
            'Before Cleaning': [df_raw[c].isnull().sum() for c in cols_present],
            'After Cleaning': [df_clean[c].isnull().sum() if c in df_clean.columns else 0 for c in cols_present]
        }).melt(id_vars='Column', var_name='Cleaning Status', value_name='Missing Count')
        
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.barplot(data=chart_data, x='Column', y='Missing Count', hue='Cleaning Status', palette='Set1', ax=ax)
        ax.set_title("Missing Values Before and After Cleaning")
        st.pyplot(fig)

# TAB 6: Correlation Heatmap
with tab6:
    st.subheader("Numerical Feature Correlation Heatmap")
    
    fig, ax = plt.subplots(figsize=(9, 5))
    num_data = filtered_clean.select_dtypes(include=[np.number])
    sns.heatmap(num_data.corr(), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation Heatmap (Cleaned Data)", fontsize=13, fontweight='bold')
    st.pyplot(fig)

st.markdown("---")
st.caption("Developed by 3rd-Year B.E. AI & Data Science Student | Titanic Missing Values Handling Minor Project")
