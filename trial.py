import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Real Estate Dashboard & Target Distribution", layout="wide")
st.title("\U0001F3D9️ Dubai Real Estate Dashboard & Target Distribution")

st.markdown(
    "[Link for the dataset to upload](https://drive.google.com/file/d/10HKlIrWIhj2TMjdFREijV_ev7hIRZXoF/view?usp=drive_link)"
)

# --- File Upload ---
uploaded_file = st.file_uploader("\U0001F4C2 Upload your CSV or Excel file", type=["csv", "xlsx"])

# --- IQR Bound Helper ---
def get_iqr_bounds(df, col):
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame for IQR calculation.")
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

# --- Box and Line Plots by Object Columns ---
def plot_box_line_by_object_column(df, target, df_name, object_col):
    st.subheader(f"\U0001F4CC Box & Line Plots by: {object_col} for {df_name}")

    if object_col not in df.columns:
        st.warning(f"Column '{object_col}' not found in {df_name}.")
        return

    fig_box = px.box(df, x=object_col, y=target, title=f'Box Plot by {object_col} ({df_name})')
    fig_box.update_layout(yaxis_title="Meter Sale Price (AED)", yaxis_tickformat=",")
    st.plotly_chart(fig_box, use_container_width=True)

    year_col = None
    for c in ['instance_year', 'instance_Year']:
        if c in df.columns:
            year_col = c
            break

    if year_col:
        grouped_data = df.groupby([year_col, object_col])[target].mean().reset_index()
        fig_line = px.line(
            grouped_data, x=year_col, y=target, color=object_col,
            title=f'Line Plot by {year_col} and {object_col} ({df_name})'
        )
        fig_line.update_layout(
            xaxis_title="Year", yaxis_title="Meter Sale Price (AED)",
            yaxis_tickformat=",", legend_title=object_col
        )
        fig_line.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig_line.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        st.plotly_chart(fig_line, use_container_width=True)

# --- Overall Target Box Plot ---
def plot_overall_target_distribution(df, target, df_name):
    st.header(f"\U0001F4CA Target Distribution Analysis for: {df_name}")

    if target not in df.columns:
        st.warning(f"Target column '{target}' not found in {df_name}. Skipping this dataset.")
        return

    fig = px.box(df, y=target, title=f'Overall Boxplot of {target} ({df_name})')
    fig.update_layout(yaxis_title="Meter Sale Price (AED)", yaxis_tickformat=",")
    st.plotly_chart(fig, use_container_width=True)

# --- Main Logic ---
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    st.subheader("\U0001F9FE Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.sidebar.header("⚙️ Configuration")
    target_column = st.sidebar.text_input("\U0001F3AF Enter the target column", value="meter_sale_price")

    tab1, tab2 = st.tabs(["\U0001F4CA Dashboard", "\U0001F3AF Target Distribution"])

    with tab1:
        st.header("\U0001F4C8 Dashboard Analysis")

        drop_cols = st.multiselect("Select columns to drop", df.columns)
        df_dash = df.drop(columns=drop_cols) if drop_cols else df.copy()

        st.subheader("\U0001F9FE Data Summary")
        st.dataframe(pd.DataFrame({
            "Column": df_dash.columns,
            "Data Type": df_dash.dtypes.astype(str),
            "Null Count": df_dash.isnull().sum(),
            "Null %": df_dash.isnull().mean().mul(100).round(2),
            "Unique Values": df_dash.nunique()
        }))

        st.subheader("\U0001F50D Univariate Analysis")
        all_cols = df_dash.columns.tolist()
        if all_cols:
            uni_col = st.selectbox("Select column", all_cols)
            plot_type = st.radio("Plot type", ["Box", "Histogram", "Line", "Bar (Freq)"], horizontal=True)

            try:
                if plot_type in ["Box", "Histogram", "Line"]:
                    if pd.api.types.is_numeric_dtype(df_dash[uni_col]):
                        if plot_type == "Box":
                            fig = px.box(df_dash, y=uni_col, title=f"Box Plot of {uni_col}")
                        elif plot_type == "Histogram":
                            fig = px.histogram(df_dash, x=uni_col, title=f"Histogram of {uni_col}")
                        elif plot_type == "Line":
                            fig = px.line(df_dash, y=uni_col, title=f"Line Plot of {uni_col}")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"'{plot_type}' plot is only supported for numeric columns.")
                elif plot_type == "Bar (Freq)":
                    freq_df = df_dash[uni_col].value_counts().reset_index()
                    freq_df.columns = [uni_col, "Count"]
                    fig = px.bar(freq_df, x=uni_col, y="Count", title=f"Frequency of {uni_col}")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ Error generating plot for '{uni_col}': {e}")

        if {'instance_year', target_column}.issubset(df_dash.columns):
            st.subheader("\U0001F4C9 Meter Sale Price Over Years")
            grouped = df_dash.groupby("instance_year")[target_column].mean().reset_index()
            counts = df_dash["instance_year"].value_counts().reset_index()
            counts.columns = ["instance_year", "Record Count"]

            fig = px.line(grouped, x="instance_year", y=target_column, title=f"{target_column.title()} Trend Over Years")
            fig.add_trace(go.Bar(x=counts["instance_year"], y=counts["Record Count"], name="Record Count", yaxis="y2"))
            fig.update_layout(
                yaxis=dict(title=target_column.title()),
                yaxis2=dict(title="Record Count", overlaying="y", side="right"),
                legend=dict(x=0, y=1),
                bargap=0.2
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("\U0001F5FA️ Interactive Area-Wise Bubble Map")
        required_map_cols = ["area_lat", "area_lon", target_column]
        if all(x in df_dash.columns for x in required_map_cols):
            df_dash["area_lat"] = pd.to_numeric(df_dash["area_lat"], errors='coerce')
            df_dash["area_lon"] = pd.to_numeric(df_dash["area_lon"], errors='coerce')
            df_dash[target_column] = pd.to_numeric(df_dash[target_column], errors='coerce')
            df_map = df_dash.dropna(subset=required_map_cols)

            if "transaction_date" in df_map.columns:
                df_map["transaction_date"] = pd.to_datetime(df_map["transaction_date"], errors='coerce')
                min_date, max_date = df_map["transaction_date"].min(), df_map["transaction_date"].max()
                if pd.isna(min_date) or pd.isna(max_date):
                    start, end = st.sidebar.date_input("Select Date Range")
                else:
                    start, end = st.sidebar.date_input("Select Date Range", [min_date, max_date])
                if start and end:
                    df_map = df_map[(df_map["transaction_date"] >= pd.to_datetime(start)) & (df_map["transaction_date"] <= pd.to_datetime(end))]

            if "area_name_en" in df_map.columns:
                areas = sorted(df_map["area_name_en"].dropna().unique())
                selected_areas = st.sidebar.multiselect("Select Areas", areas, default=areas)
                df_map = df_map[df_map["area_name_en"].isin(selected_areas)]

            grouped = df_map.groupby(["area_name_en", "area_lat", "area_lon"])[target_column].agg(["count", "mean"]).reset_index()
            grouped.columns = ["Area", "Lat", "Lon", "Record Count", "Avg Meter Price"]

            min_p, max_p = grouped["Avg Meter Price"].min(), grouped["Avg Meter Price"].max()
            price_filter = st.sidebar.slider("Avg. Meter Price Range", float(min_p), float(max_p), (float(min_p), float(max_p)))
            grouped = grouped[(grouped["Avg Meter Price"] >= price_filter[0]) & (grouped["Avg Meter Price"] <= price_filter[1])]

            fig = px.scatter_mapbox(
                grouped, lat="Lat", lon="Lon", size="Record Count", color="Avg Meter Price",
                hover_name="Area", size_max=50, zoom=10, mapbox_style="open-street-map",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### \U0001F4E5 Download Filtered Map Data")
            st.download_button(
                label="⬇️ Download CSV",
                data=grouped.to_csv(index=False),
                file_name="filtered_map_data.csv",
                mime="text/csv"
            )
        else:
            st.info(f"Map requires columns: {', '.join(required_map_cols)}")

    with tab2:
        st.header("\U0001F3AF Comparative Target Distribution Dashboard")

        try:
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

            if st.button("\U0001F4CA Generate Target Distribution Plots"):
                for i, d in enumerate(dfs):
                    plot_overall_target_distribution(d, target_column, df_names[i])

                object_cols = [
                    'trans_group_en', 'procedure_name_en', 'property_type_en', 'property_sub_type_en',
                    'property_usage_en', 'reg_type_en', 'nearest_landmark_en', 'nearest_metro_en',
                    'nearest_mall_en', 'rooms_en'
                ]

                for col in object_cols:
                    for i, d in enumerate(dfs):
                        plot_box_line_by_object_column(d, target_column, df_names[i], col)

        except KeyError as ke:
            st.error(f"Missing column for IQR filtering: {ke}")
        except Exception as e:
            st.error(f"❌ Error during IQR filtering or plotting: {e}")
else:
    st.info("👈 Upload a CSV or Excel file to begin analysis.")
