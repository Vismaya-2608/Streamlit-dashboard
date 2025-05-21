import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Real Estate Dashboard & Target Distribution", layout="wide")
st.title("🏙️ Dubai Real Estate Dashboard & Target Distribution")

st.markdown(
    "[Link for the dataset to upload](https://drive.google.com/file/d/10HKlIrWIhj2TMjdFREijV_ev7hIRZXoF/view?usp=drive_link)"
)

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload your CSV or Excel file", type=["csv", "xlsx"])

# --- IQR Bound Helper ---
def get_iqr_bounds(df, col):
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame for IQR calculation.")
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr
  
def plot_target_distribution_by_object_columns_streamlit(dfs, target_column, df_names):
    object_cols = [
    'trans_group_en', 'procedure_name_en', 'property_type_en', 'property_sub_type_en',
    'property_usage_en', 'reg_type_en', 'nearest_landmark_en', 'nearest_metro_en',
    'nearest_mall_en', 'rooms_en'
]

# Plot overall boxplots
    for i, df_ in enumerate(dfs):
    df_name = df_names[i]
    st.header(f"📊 Target Distribution Analysis for: {df_name}")

    if target_column not in df_.columns:
        st.warning(f"Target column '{target_column}' not found in {df_name}. Skipping this dataset.")
        continue

    fig = px.box(df_, y=target_column, title=f'Overall Boxplot of {target_column} ({df_name})')
    fig.update_layout(yaxis_title="Meter Sale Price (AED)", yaxis_tickformat=",")
    st.plotly_chart(fig, use_container_width=True)

# Box and line plots by object columns
    for col in object_cols:
    st.subheader(f"📌 Box & Line Plots by: {col}")
    cols = st.columns(len(dfs))

    for i, df_ in enumerate(dfs):
        df_name = df_names[i]
        if col not in df_.columns:
            continue

        with cols[i]:
            st.markdown(f"**{df_name}**")

            fig_box = px.box(df_, x=col, y=target_column, title=f'Box Plot by {col} ({df_name})')
            fig_box.update_layout(yaxis_title="Meter Sale Price (AED)", yaxis_tickformat=",")
            st.plotly_chart(fig_box, use_container_width=True)

            # Year column detection
            year_col = None
            for c in ['instance_year', 'instance_Year', 'instance_year']:  # typo safety
                if c in df_.columns:
                    year_col = c
                    break

            if year_col is not None:
                grouped_data = df_.groupby([year_col, col])[target_column].mean().reset_index()
                fig_line = px.line(
                    grouped_data, x=year_col, y=target_column, color=col,
                    title=f'Line Plot by {year_col} and {col} ({df_name})'
                )
                fig_line.update_layout(
                    xaxis_title="Year", yaxis_title="Meter Sale Price (AED)",
                    yaxis_tickformat=",", legend_title=col
                )
                fig_line.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                fig_line.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                st.plotly_chart(fig_line, use_container_width=True)
    st.markdown("---")
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Normalize column names: lowercase and underscores
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    st.subheader("🧾 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

try:
            # Check if required columns exist for IQR filtering
            for col in ['meter_sale_price', 'procedure_area']:
                if col not in df.columns:
                    st.warning(f"Column '{col}' missing for outlier filtering. Skipping filtering.")
                    raise KeyError(col)

            mlower, mupper = get_iqr_bounds(df, 'meter_sale_price')
            plower, pupper = get_iqr_bounds(df, 'procedure_area')

            otdf = df[(df['meter_sale_price'] >= mlower) & (df['meter_sale_price'] <= mupper)]
            odf = otdf[(otdf['procedure_area'] >= plower) & (otdf['procedure_area'] <= pupper)]

            dfs = [df, odf]
            df_names = ['Raw Data', 'Data after Cleaning Outliers']

            if st.button("📊 Generate Target Distribution Plots"):
                st.success(f"Generating plots for target column: **{target_column}**")
                plot_target_distribution_by_object_columns_streamlit(dfs, target_column, df_names)

except KeyError as ke:
            st.error(f"Missing column for IQR filtering: {ke}")
except Exception as e:
            st.error(f"❌ Error during IQR filtering or plotting: {e}")
else:
    st.info("👈 Upload a CSV or Excel file to begin analysis.")




