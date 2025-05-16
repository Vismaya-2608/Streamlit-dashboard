import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Streamlit configuration
st.set_page_config(page_title="Data Dashboard", layout="wide")
st.title("📊 Data Analysis Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Read file based on extension
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"{uploaded_file.name} loaded successfully!")

        # === RAW DATA PREVIEW ===
        st.subheader("📁 Raw Data Preview")
        st.dataframe(df.head())

        # === BASIC SUMMARY ===
        st.subheader("📌 Data Summary")
        summary = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": df.dtypes.values,
            "Missing Values": df.isnull().sum().values,
            "Missing %": round(df.isnull().mean() * 100, 2)
        })
        st.dataframe(summary)

        # === SUMMARY STATISTICS ===
        st.subheader("📈 Summary Statistics")
        st.write(df.describe(include='all').T)

        # === UNIVARIATE ANALYSIS ===
        st.subheader("📊 Univariate Analysis")
        col_uni = st.selectbox("Select a column for univariate analysis", df.columns)

        if pd.api.types.is_numeric_dtype(df[col_uni]):
            fig = px.histogram(df, x=col_uni, nbins=30, title=f"Distribution of {col_uni}")
            st.plotly_chart(fig)
        else:
            cat_counts = df[col_uni].value_counts().reset_index()
            cat_counts.columns = [col_uni, 'Count']  # Rename properly

            fig = px.bar(cat_counts, x=col_uni, y='Count',
             labels={col_uni: col_uni, 'Count': 'Count'},
             title=f"Count plot of {col_uni}")

            
            st.plotly_chart(fig)

        # === BIVARIATE ANALYSIS ===
        st.subheader("📉 Bivariate Analysis")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(num_cols) >= 2:
            col_x = st.selectbox("Select X-axis column", num_cols, index=0)
            col_y = st.selectbox("Select Y-axis column", num_cols, index=1)

            fig2 = px.scatter(df, x=col_x, y=col_y, title=f"{col_x} vs {col_y}")
            st.plotly_chart(fig2)

            st.subheader("🔍 Correlation Heatmap")
            corr = df[num_cols].corr()
            fig3, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig3)
        else:
            st.info("Need at least two numerical columns for bivariate analysis.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a CSV or Excel file to get started.")
