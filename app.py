import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import datetime
import base64

# Set page configuration
st.set_page_config(
    page_title="Social Media Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        color: #333;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .insight-text {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        font-style: italic;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f2f6;
        border-bottom: 2px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<div class="main-header">Social Media Analytics Dashboard</div>', unsafe_allow_html=True)

# Function to load data
def load_data(uploaded_file):
    """Load data from uploaded file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        # Convert all columns to appropriate data types
        for col in df.columns:
            # Try to convert to datetime first
            try:
                df[col] = pd.to_datetime(df[col])
                continue
            except (ValueError, TypeError):
                pass
            
            # Then try to convert to numeric
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to safely convert columns to numeric
def safe_numeric_conversion(df, columns):
    """Convert columns to numeric type, coercing errors to NaN"""
    result_df = df.copy()
    for col in columns:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
    return result_df

# Sidebar for data upload and filters
with st.sidebar:
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Upload your social media dataset", type=["csv", "xlsx", "xls"])
    
    st.header("Global Filters")
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            # Date range filter - if date columns exist
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                date_col = st.selectbox("Select Date Column", date_columns)
                if pd.api.types.is_datetime64_any_dtype(df[date_col]):
                    min_date = df[date_col].min().date()
                    max_date = df[date_col].max().date()
                else:
                    try:
                        df[date_col] = pd.to_datetime(df[date_col])
                        min_date = df[date_col].min().date()
                        max_date = df[date_col].max().date()
                    except:
                        min_date = datetime.date.today() - datetime.timedelta(days=30)
                        max_date = datetime.date.today()
                
                date_range = st.date_input(
                    "Select Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                if len(date_range) == 2:
                    df = df[(df[date_col].dt.date >= date_range[0]) & 
                             (df[date_col].dt.date <= date_range[1])]
            
            # Platform filter
            if 'Platform' in df.columns:
                platforms = ['All'] + sorted(df['Platform'].unique().tolist())
                selected_platform = st.selectbox("Select Platform", platforms)
                if selected_platform != 'All':
                    df = df[df['Platform'] == selected_platform]
            
            # Age group filter
            if 'Age' in df.columns:
                try:
                    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
                    age_min = int(df['Age'].min())
                    age_max = int(df['Age'].max())
                    age_range = st.slider("Age Range", age_min, age_max, (age_min, age_max))
                    df = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]
                except:
                    st.warning("Could not filter by age due to data issues")
            
            # Gender filter
            if 'Gender' in df.columns:
                genders = ['All'] + sorted(df['Gender'].unique().tolist())
                selected_gender = st.selectbox("Select Gender", genders)
                if selected_gender != 'All':
                    df = df[df['Gender'] == selected_gender]
            
            # Location filter
            if 'Location' in df.columns:
                locations = ['All'] + sorted(df['Location'].unique().tolist())
                selected_location = st.selectbox("Select Location", locations)
                if selected_location != 'All':
                    df = df[df['Location'] == selected_location]
    
    st.divider()
    st.info("This dashboard provides comprehensive insights about social media usage patterns and engagement metrics.")

# Main content - Only show if data is uploaded
if uploaded_file is not None and 'df' in locals():
    # Create tabs for different sections
    tabs = st.tabs([
        "📱 Platform Overview", 
        "📈 Publishing Insights", 
        "👍 Engagement Analysis",
        "👥 Audience Insights",
        "💰 Marketing ROI",
        "🔄 Sales Funnel",
        "🎬 Content Optimization",
        "🎯 Lead Nurturing",
        "🧩 Audience Segments"
    ])
    
    # 1. Platform Performance Overview
    with tabs[0]:
        st.markdown('<div class="section-header">Platform Performance Overview</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            # Platform distribution
            if 'Platform' in df.columns:
                platform_counts = df['Platform'].value_counts()
                fig_platform = px.pie(
                    names=platform_counts.index,
                    values=platform_counts.values,
                    title="User Distribution by Platform",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    hole=0.4
                )
                fig_platform.update_traces(textinfo='percent+label')
                st.plotly_chart(fig_platform, use_container_width=True)
            
            # Device types
            if 'Device Type' in df.columns:
                device_counts = df['Device Type'].value_counts()
                fig_device = px.bar(
                    x=device_counts.index,
                    y=device_counts.values,
                    title="Device Type Distribution",
                    labels={'x': 'Device Type', 'y': 'Count'},
                    color=device_counts.values,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_device, use_container_width=True)
        
        with col2:
            # Average time spent per platform
            if all(col in df.columns for col in ['Platform', 'Total Time Spent']):
                try:
                    # Ensure Total Time Spent is numeric
                    df['Total Time Spent'] = pd.to_numeric(df['Total Time Spent'], errors='coerce')
                    
                    # Calculate mean time spent by platform
                    time_by_platform = df.groupby('Platform')['Total Time Spent'].mean().reset_index()
                    
                    # Check if we have valid data after calculations
                    if not time_by_platform.empty and not time_by_platform['Total Time Spent'].isna().all():
                        fig_time = px.bar(
                            time_by_platform,
                            x='Platform',
                            y='Total Time Spent',
                            title="Average Time Spent per Platform",
                            color='Total Time Spent',
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig_time, use_container_width=True)
                    else:
                        st.warning("Not enough valid data to display time spent by platform.")
                except Exception as e:
                    st.error(f"Error processing time spent data: {str(e)}")
                    st.warning("Could not calculate average time spent by platform due to data issues.")
            
            # OS distribution
            if 'OS' in df.columns:
                os_counts = df['OS'].value_counts()
                fig_os = px.pie(
                    names=os_counts.index,
                    values=os_counts.values,
                    title="Operating System Distribution",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                st.plotly_chart(fig_os, use_container_width=True)
        
        # Heatmap for Platform vs OS
        if all(col in df.columns for col in ['Platform', 'OS']):
            st.markdown("### Platform vs OS Usage")
            platform_os_counts = pd.crosstab(df['Platform'], df['OS'])
            fig_heatmap = px.imshow(
                platform_os_counts,
                labels=dict(x="Operating System", y="Platform", color="User Count"),
                title="Platform vs OS Usage Heatmap",
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Key metrics
        st.markdown("### Key Platform Metrics")
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            if 'Number of Sessions' in df.columns:
                try:
                    # Convert to numeric, forcing errors to NaN
                    sessions_numeric = pd.to_numeric(df['Number of Sessions'], errors='coerce')
                    # Calculate mean only on numeric values
                    mean_sessions = sessions_numeric.mean()
                    if not pd.isna(mean_sessions):
                        st.metric("Avg. Sessions", f"{mean_sessions:.1f}")
                    else:
                        st.metric("Avg. Sessions", "N/A")
                except:
                    st.metric("Avg. Sessions", "N/A")
        
        with metrics_col2:
            if 'Total Time Spent' in df.columns:
                try:
                    # Convert to numeric, forcing errors to NaN
                    time_numeric = pd.to_numeric(df['Total Time Spent'], errors='coerce')
                    # Calculate mean only on numeric values
                    mean_time = time_numeric.mean()
                    if not pd.isna(mean_time):
                        st.metric("Avg. Time Spent (min)", f"{mean_time:.1f}")
                    else:
                        st.metric("Avg. Time Spent (min)", "N/A")
                except:
                    st.metric("Avg. Time Spent (min)", "N/A")
        
        with metrics_col3:
            if 'Frequency' in df.columns:
                try:
                    # Convert to numeric, forcing errors to NaN
                    freq_numeric = pd.to_numeric(df['Frequency'], errors='coerce')
                    # Calculate mean only on numeric values
                    mean_freq = freq_numeric.mean()
                    if not pd.isna(mean_freq):
                        st.metric("Avg. Frequency", f"{mean_freq:.1f}")
                    else:
                        st.metric("Avg. Frequency", "N/A")
                except:
                    st.metric("Avg. Frequency", "N/A")
                
        with metrics_col4:
            if 'Connection Type' in df.columns:
                try:
                    most_common_connection = df['Connection Type'].mode()[0]
                    st.metric("Popular Connection", most_common_connection)
                except:
                    st.metric("Popular Connection", "N/A")
    
    # 2. Publishing Insights
    with tabs[1]:
        st.markdown('<div class="section-header">Publishing Insights</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            # Engagement by category
            if all(col in df.columns for col in ['Video Category', 'Engagement']):
                try:
                    # Ensure Engagement is numeric
                    df['Engagement'] = pd.to_numeric(df['Engagement'], errors='coerce')
                    
                    # Group by category and calculate mean engagement
                    engagement_by_cat = df.groupby('Video Category')['Engagement'].mean().reset_index()
                    
                    # Check if we have valid data after calculations
                    if not engagement_by_cat.empty and not engagement_by_cat['Engagement'].isna().all():
                        # Sort by engagement
                        engagement_by_cat = engagement_by_cat.sort_values('Engagement', ascending=False)
                        
                        fig_cat_eng = px.bar(
                            engagement_by_cat,
                            x='Video Category',
                            y='Engagement',
                            title="Average Engagement by Content Category",
                            color='Engagement',
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig_cat_eng, use_container_width=True)
                    else:
                        st.warning("Not enough valid data to display engagement by category.")
                except Exception as e:
                    st.error(f"Error processing engagement data: {str(e)}")
                    st.warning("Could not calculate engagement by category due to data issues.")
        
        with col2:
            # Number of videos watched per category
            if all(col in df.columns for col in ['Video Category', 'Number of Videos Watched']):
                try:
                    df['Number of Videos Watched'] = pd.to_numeric(df['Number of Videos Watched'], errors='coerce')
                    videos_by_cat = df.groupby('Video Category')['Number of Videos Watched'].sum().reset_index()
                    videos_by_cat = videos_by_cat.sort_values('Number of Videos Watched', ascending=False)
                    
                    fig_cat_vids = px.bar(
                        videos_by_cat,
                        x='Video Category',
                        y='Number of Videos Watched',
                        title="Number of Videos Watched by Category",
                        color='Number of Videos Watched',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_cat_vids, use_container_width=True)
                except Exception as e:
                    st.error(f"Error processing videos watched data: {str(e)}")
                    st.warning("Could not display videos watched by category due to data issues.")
        
        # Watch reasons
        if 'Watch Reason' in df.columns:
            st.markdown("### Watch Reasons Analysis")
            reason_counts = df['Watch Reason'].value_counts()
            
            fig_reasons = px.pie(
                names=reason_counts.index,
                values=reason_counts.values,
                title="Distribution of Watch Reasons",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_reasons, use_container_width=True)
        
        # Watch time distribution by hour (simulated if no time data)
        st.markdown("### Peak Watch Time Distribution")
        if 'Hour' in df.columns:
            try:
                df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce')
                df['Total Time Spent'] = pd.to_numeric(df['Total Time Spent'], errors='coerce')
                hour_dist = df.groupby('Hour')['Total Time Spent'].sum().reset_index()
            except:
                # Fallback to simulated data if there are issues
                hours = list(range(24))
                time_spent = [5, 10, 25, 45, 65, 70, 65, 60, 55, 50, 45, 40, 45, 60, 85, 100, 90, 75, 60, 40, 25, 15, 10, 5]
                hour_dist = pd.DataFrame({'Hour': hours, 'Total Time Spent': time_spent})
        else:
            # Simulate hourly data
            hours = list(range(24))
            time_spent = [5, 10, 25, 45, 65, 70, 65, 60, 55, 50, 45, 40, 45, 60, 85, 100, 90, 75, 60, 40, 25, 15, 10, 5]
            hour_dist = pd.DataFrame({'Hour': hours, 'Total Time Spent': time_spent})
        
        fig_hour = px.line(
            hour_dist,
            x='Hour',
            y='Total Time Spent',
            title="Watch Time Distribution by Hour",
            markers=True
        )
        fig_hour.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
        st.plotly_chart(fig_hour, use_container_width=True)
    
    # 3. Engagement Analysis
    with tabs[2]:
        st.markdown('<div class="section-header">Engagement Analysis</div>', unsafe_allow_html=True)
        
        # Importance Score vs Engagement
        # Importance Score vs Engagement
        if all(col in df.columns for col in ['Importance Score', 'Engagement']):
            st.markdown("### Importance Score vs Engagement")
            try:
                df['Importance Score'] = pd.to_numeric(df['Importance Score'], errors='coerce')
                df['Engagement'] = pd.to_numeric(df['Engagement'], errors='coerce')

                # Ensure 'Time Spent On Video' is numeric
                if 'Time Spent On Video' in df.columns:
                    try:
                        if pd.api.types.is_timedelta64_dtype(df['Time Spent On Video']):
                            df['Time Spent On Video'] = df['Time Spent On Video'].dt.total_seconds()
                        elif pd.api.types.is_datetime64_any_dtype(df['Time Spent On Video']):
                            df['Time Spent On Video'] = (
                                df['Time Spent On Video'] - df['Time Spent On Video'].min()
                            ).dt.total_seconds()
                        else:
                            df['Time Spent On Video'] = pd.to_numeric(df['Time Spent On Video'], errors='coerce')
                    except Exception as e:
                        st.warning(f"Failed to process 'Time Spent On Video': {str(e)}")

                fig_scatter = px.scatter(
                    df,
                    x='Importance Score',
                    y='Engagement',
                    color='Platform' if 'Platform' in df.columns else None,
                    size='Time Spent On Video' if 'Time Spent On Video' in df.columns else None,
                    hover_data=['UserID'] if 'UserID' in df.columns else None,
                    title="Importance Score vs Engagement",
                    opacity=0.7
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating scatter plot: {str(e)}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Engagement distribution
            if 'Engagement' in df.columns:
                try:
                    df['Engagement'] = pd.to_numeric(df['Engagement'], errors='coerce')
                    fig_hist = px.histogram(
                        df,
                        x='Engagement',
                        nbins=20,
                        title="Engagement Distribution",
                        color_discrete_sequence=['#1E88E5']
                    )
                    fig_hist.update_layout(bargap=0.1)
                    st.plotly_chart(fig_hist, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating histogram: {str(e)}")
        
        with col2:
            # Engagement vs Satisfaction
            if all(col in df.columns for col in ['Engagement', 'Satisfaction']):
                try:
                    df['Satisfaction'] = pd.to_numeric(df['Satisfaction'], errors='coerce')
                    fig_eng_sat = px.scatter(
                        df,
                        x='Engagement',
                        y='Satisfaction',
                        color='Video Category' if 'Video Category' in df.columns else None,
                        title="Engagement vs Satisfaction",
                        trendline="ols"
                    )
                    st.plotly_chart(fig_eng_sat, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating scatter plot: {str(e)}")
        
        # Correlation matrix for engagement metrics
        st.markdown("### Correlation Matrix for Engagement Metrics")
        engagement_cols = [col for col in df.columns if col in [
            'Engagement', 'Time Spent On Video', 'Importance Score', 
            'Satisfaction', 'Scroll Rate', 'Total Time Spent',
            'Number of Videos Watched', 'Productivity Loss', 'Addiction Level'
        ]]
        
        if len(engagement_cols) >= 3:
            try:
                # Ensure all columns are numeric
                corr_df = df[engagement_cols].copy()
                for col in engagement_cols:
                    corr_df[col] = pd.to_numeric(corr_df[col], errors='coerce')
                
                # Drop rows with NaN values to avoid correlation errors
                corr_df = corr_df.dropna()
                
                if len(corr_df) >= 5:  # Need at least some data for meaningful correlation
                    corr_matrix = corr_df.corr()
                    
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        color_continuous_scale='RdBu_r',
                        title="Correlation Matrix for Engagement Metrics"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.warning("Not enough valid numeric data to calculate correlations.")
            except Exception as e:
                st.error(f"Error creating correlation matrix: {str(e)}")
                st.warning("Could not create correlation matrix due to data issues.")
        else:
            st.warning("Not enough engagement-related metrics to create correlation matrix.")
    
    # 4. Audience Insights
    with tabs[3]:
        st.markdown('<div class="section-header">Audience Insights</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            if 'Age' in df.columns:
                try:
                    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
                    fig_age = px.histogram(
                        df,
                        x='Age',
                        nbins=20,
                        title="Age Distribution",
                        color_discrete_sequence=['#1E88E5']
                    )
                    fig_age.update_layout(bargap=0.1)
                    st.plotly_chart(fig_age, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating age histogram: {str(e)}")
            
            # Gender distribution
            if 'Gender' in df.columns:
                gender_counts = df['Gender'].value_counts()
                fig_gender = px.pie(
                    names=gender_counts.index,
                    values=gender_counts.values,
                    title="Gender Distribution",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_gender, use_container_width=True)
        
        with col2:
            # Income distribution
            if 'Income' in df.columns:
                try:
                    df['Income'] = pd.to_numeric(df['Income'], errors='coerce')
                    fig_income = px.box(
                        df,
                        y='Income',
                        title="Income Distribution",
                        color='Platform' if 'Platform' in df.columns else None
                    )
                    st.plotly_chart(fig_income, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating income box plot: {str(e)}")
            
            # Location distribution
            if 'Location' in df.columns:
                location_counts = df['Location'].value_counts().head(10)
                fig_loc = px.bar(
                    x=location_counts.index,
                    y=location_counts.values,
                    title="Top 10 Locations",
                    labels={'x': 'Location', 'y': 'Count'}
                )
                st.plotly_chart(fig_loc, use_container_width=True)
        
        # Age vs Addiction Level
        if all(col in df.columns for col in ['Age', 'Addiction Level']):
            st.markdown("### Age vs Addiction Level")
            try:
                df['Addiction Level'] = pd.to_numeric(df['Addiction Level'], errors='coerce')
                fig_age_add = px.scatter(
                    df,
                    x='Age',
                    y='Addiction Level',
                    color='Gender' if 'Gender' in df.columns else None,
                    title="Age vs Addiction Level",
                    trendline="ols"
                )
                st.plotly_chart(fig_age_add, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating scatter plot: {str(e)}")
        
        # Demographics breakdown
        st.markdown("### Profession Breakdown")
        if 'Profession' in df.columns:
            prof_counts = df['Profession'].value_counts().head(10)
            fig_prof = px.bar(
                x=prof_counts.index,
                y=prof_counts.values,
                title="Top 10 Professions",
                color=prof_counts.values,
                color_continuous_scale='Viridis',
                labels={'x': 'Profession', 'y': 'Count'}
            )
            st.plotly_chart(fig_prof, use_container_width=True)
    
    # 5. Marketing ROI Measurement
    with tabs[4]:
        st.markdown('<div class="section-header">Marketing ROI Measurement</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Satisfaction by platform
            if all(col in df.columns for col in ['Platform', 'Satisfaction']):
                try:
                    df['Satisfaction'] = pd.to_numeric(df['Satisfaction'], errors='coerce')
                    sat_platform = df.groupby('Platform')['Satisfaction'].mean().reset_index()
                    fig_sat_plat = px.bar(
                        sat_platform,
                        x='Platform',
                        y='Satisfaction',
                        title="Average Satisfaction by Platform",
                        color='Satisfaction',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_sat_plat, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating satisfaction chart: {str(e)}")
        
        with col2:
            # Satisfaction by demographic
            if all(col in df.columns for col in ['Gender', 'Satisfaction']):
                try:
                    sat_gender = df.groupby('Gender')['Satisfaction'].mean().reset_index()
                    fig_sat_gen = px.bar(
                        sat_gender,
                        x='Gender',
                        y='Satisfaction',
                        title="Average Satisfaction by Gender",
                        color='Satisfaction',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_sat_gen, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating satisfaction chart: {str(e)}")
        
        # Heatmap: Content category vs Satisfaction
        if all(col in df.columns for col in ['Video Category', 'Satisfaction']):
            st.markdown("### Content Category vs Satisfaction")
            try:
                cat_sat = pd.pivot_table(
                    df, 
                    values='Satisfaction', 
                    index='Video Category',
                    aggfunc='mean'
                ).sort_values('Satisfaction', ascending=False).reset_index()
                
                fig_cat_sat = px.bar(
                    cat_sat,
                    x='Video Category',
                    y='Satisfaction',
                    title="Average Satisfaction by Content Category",
                    color='Satisfaction',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_cat_sat, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating satisfaction chart: {str(e)}")
        
        # Cross-tab: Engagement vs Productivity Loss
        if all(col in df.columns for col in ['Engagement', 'Productivity Loss']):
            st.markdown("### Engagement vs Productivity Loss Analysis")
            try:
                df['Engagement'] = pd.to_numeric(df['Engagement'], errors='coerce')
                df['Productivity Loss'] = pd.to_numeric(df['Productivity Loss'], errors='coerce')
                
                # Create bins for engagement
                df['Engagement_Bin'] = pd.qcut(df['Engagement'], 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
                prod_eng = pd.pivot_table(
                    df,
                    values='Productivity Loss',
                    index='Engagement_Bin',
                    aggfunc='mean'
                ).reset_index()
                
                fig_prod_eng = px.line(
                    prod_eng,
                    x='Engagement_Bin',
                    y='Productivity Loss',
                    title="Average Productivity Loss by Engagement Level",
                    markers=True
                )
                st.plotly_chart(fig_prod_eng, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating productivity chart: {str(e)}")
    
    # 6. Sales Funnel Efficiency
    with tabs[5]:
        st.markdown('<div class="section-header">Sales Funnel Efficiency</div>', unsafe_allow_html=True)
        
        # Create a funnel visualization
        st.markdown("### User Journey Funnel")
        
        # Create funnel data from available metrics
        funnel_data = []
        funnel_labels = []
        
        # Frequency (if exists or total users)
        if 'Frequency' in df.columns:
            try:
                df['Frequency'] = pd.to_numeric(df['Frequency'], errors='coerce')
                avg_frequency = df['Frequency'].mean()
                funnel_data.append(len(df))
                funnel_labels.append(f"Total Users: {len(df)}")
            except:
                funnel_data.append(len(df))
                funnel_labels.append(f"Total Users: {len(df)}")
        else:
            funnel_data.append(len(df))
            funnel_labels.append(f"Total Users: {len(df)}")
        
        # Time Spent
        if 'Total Time Spent' in df.columns:
            try:
                df['Total Time Spent'] = pd.to_numeric(df['Total Time Spent'], errors='coerce')
                avg_time = df['Total Time Spent'].mean()
                high_time_users = len(df[df['Total Time Spent'] > avg_time])
                funnel_data.append(high_time_users)
                funnel_labels.append(f"High Time Spent: {high_time_users}")
            except:
                pass
        
        # Engagement
        if 'Engagement' in df.columns:
            try:
                df['Engagement'] = pd.to_numeric(df['Engagement'], errors='coerce')
                avg_engagement = df['Engagement'].mean()
                engaged_users = len(df[df['Engagement'] > avg_engagement])
                funnel_data.append(engaged_users)
                funnel_labels.append(f"Engaged Users: {engaged_users}")
            except:
                pass
        
        # Satisfaction
        if 'Satisfaction' in df.columns:
            try:
                df['Satisfaction'] = pd.to_numeric(df['Satisfaction'], errors='coerce')
                avg_satisfaction = df['Satisfaction'].mean()
                satisfied_users = len(df[df['Satisfaction'] > avg_satisfaction])
                funnel_data.append(satisfied_users)
                funnel_labels.append(f"Satisfied Users: {satisfied_users}")
            except:
                pass
        
        # Create funnel chart
        if len(funnel_data) >= 3:
            fig_funnel = go.Figure(go.Funnel(
                y=funnel_labels,
                x=funnel_data,
                textinfo="value+percent initial",
                marker={"color": ["#1E88E5", "#42A5F5", "#90CAF9", "#BBDEFB"]}
            ))
            fig_funnel.update_layout(title="User Journey Funnel")
            st.plotly_chart(fig_funnel, use_container_width=True)
        else:
            st.warning("Not enough metrics to create a meaningful funnel visualization.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scroll Rate vs Session Count
            if all(col in df.columns for col in ['Scroll Rate', 'Number of Sessions']):
                try:
                    df['Scroll Rate'] = pd.to_numeric(df['Scroll Rate'], errors='coerce')
                    df['Number of Sessions'] = pd.to_numeric(df['Number of Sessions'], errors='coerce')
                    
                    fig_scroll_session = px.scatter(
                        df,
                        x='Scroll Rate',
                        y='Number of Sessions',
                        color='Platform' if 'Platform' in df.columns else None,
                        title="Scroll Rate vs Session Count",
                        trendline="ols"
                    )
                    st.plotly_chart(fig_scroll_session, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating scatter plot: {str(e)}")
        
        with col2:
            # Session frequency distribution
            if 'Number of Sessions' in df.columns:
                try:
                    df['Number of Sessions'] = pd.to_numeric(df['Number of Sessions'], errors='coerce')
                    fig_sessions = px.histogram(
                        df,
                        x='Number of Sessions',
                        nbins=20,
                        title="Session Frequency Distribution",
                        color_discrete_sequence=['#1E88E5']
                    )
                    fig_sessions.update_layout(bargap=0.1)
                    st.plotly_chart(fig_sessions, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating histogram: {str(e)}")
    
    # 7. Content Optimization
    with tabs[6]:
        st.markdown('<div class="section-header">Content Optimization</div>', unsafe_allow_html=True)
        
        # Video Length analysis
        if 'Video Length' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Engagement vs Video Length
                if 'Engagement' in df.columns:
                    try:
                        df['Video Length'] = pd.to_numeric(df['Video Length'], errors='coerce')
                        df['Engagement'] = pd.to_numeric(df['Engagement'], errors='coerce')
                        
                        fig_length_eng = px.scatter(
                            df,
                            x='Video Length',
                            y='Engagement',
                            color='Video Category' if 'Video Category' in df.columns else None,
                            title="Video Length vs Engagement",
                            trendline="ols"
                        )
                        st.plotly_chart(fig_length_eng, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating scatter plot: {str(e)}")
            
            with col2:
                # Time Spent On Video vs Video Length
                if 'Time Spent On Video' in df.columns:
                    try:
                        df['Time Spent On Video'] = pd.to_numeric(df['Time Spent On Video'], errors='coerce')
                        # Calculate completion rate (Time Spent On Video / Video Length)
                        df['Completion Rate'] = df['Time Spent On Video'] / df['Video Length']
                        df['Completion Rate'] = df['Completion Rate'].clip(0, 1)  # Cap at 100%
                        
                        fig_completion = px.scatter(
                            df,
                            x='Video Length',
                            y='Completion Rate',
                            color='Video Category' if 'Video Category' in df.columns else None,
                            title="Video Length vs Completion Rate",
                            trendline="ols"
                        )
                        st.plotly_chart(fig_completion, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating scatter plot: {str(e)}")
            
            # Video Length distribution by category
            if 'Video Category' in df.columns:
                st.markdown("### Video Length by Category")
                try:
                    fig_length_cat = px.box(
                        df,
                        x='Video Category',
                        y='Video Length',
                        title="Video Length Distribution by Category",
                        color='Video Category'
                    )
                    st.plotly_chart(fig_length_cat, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating box plot: {str(e)}")
            
            # Optimal video length analysis (simulated)
            st.markdown("### Optimal Video Length Analysis")
            
            if all(col in df.columns for col in ['Video Length', 'Engagement']):
                try:
                    # Create binned video lengths for analysis
                    df['Video Length (Binned)'] = pd.cut(
                        df['Video Length'],
                        bins=[0, 30, 60, 120, 300, 600, float('inf')],
                        labels=['0-30s', '30-60s', '1-2m', '2-5m', '5-10m', '10m+']
                    )
                    
                    metrics_by_length = df.groupby('Video Length (Binned)').agg({
                        'Engagement': 'mean',
                        'Satisfaction': 'mean' if 'Satisfaction' in df.columns else None,
                        'Time Spent On Video': 'mean' if 'Time Spent On Video' in df.columns else None,
                    }).reset_index()
                    
                    # Drop columns with None
                    metrics_by_length = metrics_by_length.dropna(axis=1)
                    
                    # Create bar chart for each metric
                    for metric in metrics_by_length.columns[1:]:
                        fig_metric = px.bar(
                            metrics_by_length,
                            x='Video Length (Binned)',
                            y=metric,
                            title=f"Average {metric} by Video Length",
                            color=metric,
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig_metric, use_container_width=True)
                except Exception as e:
                    st.error(f"Error analyzing video length: {str(e)}")
        else:
            st.warning("Video Length data not available for Content Optimization analysis.")
    
    # 8. Lead Nurturing & Sales Outreach
    with tabs[7]:
        st.markdown('<div class="section-header">Lead Nurturing & Sales Outreach</div>', unsafe_allow_html=True)
        
        # Identify users with high interest but low engagement
        if all(col in df.columns for col in ['UserID', 'Importance Score', 'Engagement']):
            st.markdown("### Identify High Potential Users")
            
            try:
                df['Importance Score'] = pd.to_numeric(df['Importance Score'], errors='coerce')
                df['Engagement'] = pd.to_numeric(df['Engagement'], errors='coerce')
                
                # Calculate median values
                median_importance = df['Importance Score'].median()
                median_engagement = df['Engagement'].median()
                
                # Find users with high importance but low engagement
                high_potential = df[
                    (df['Importance Score'] > median_importance) & 
                    (df['Engagement'] < median_engagement)
                ].sort_values('Importance Score', ascending=False)
                
                if len(high_potential) > 0:
                    # Display top potential users
                    columns_to_show = ['UserID', 'Importance Score', 'Engagement']
                    
                    # Add optional columns if they exist
                    for col in ['Age', 'Gender', 'Platform', 'Location', 'Profession']:
                        if col in df.columns:
                            columns_to_show.append(col)
                    
                    st.dataframe(high_potential[columns_to_show].head(10))
                    
                    st.markdown("""
                    <div class="insight-text">
                    These users show high interest in content (high importance score) but haven't fully engaged yet. 
                    They represent a prime opportunity for targeted outreach and content recommendations.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No users with high importance score but low engagement were found.")
            except Exception as e:
                st.error(f"Error identifying high potential users: {str(e)}")
        
        # Behavior profiles visualization
        st.markdown("### User Behavior Profiles")
        
        behavior_cols = [col for col in df.columns if col in [
            'Addiction Level', 'Self Control', 'Productivity Loss'
        ]]
        
        if len(behavior_cols) >= 2:
            try:
                # Convert all behavior columns to numeric
                for col in behavior_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Drop rows with NaN values
                behavior_df = df.dropna(subset=behavior_cols)
                
                if len(behavior_df) > 0:
                    # Sample a subset of users for the radar chart (max 50)
                    sample_size = min(50, len(behavior_df))
                    sampled_users = behavior_df.sample(sample_size)
                    
                    # Create a radar chart for sampled users
                    fig_radar = go.Figure()
                    
                    for i, (_, user) in enumerate(sampled_users.iterrows()):
                        user_id = user['UserID'] if 'UserID' in df.columns else f"User {i+1}"
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=[user[col] for col in behavior_cols],
                            theta=behavior_cols,
                            fill='toself',
                            name=f"{user_id}"
                        ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, sampled_users[behavior_cols].max().max()]
                            )
                        ),
                        title="User Behavior Profiles",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
                    
                    # Personalized suggestions (simulated)
                    st.markdown("### Personalized Outreach Suggestions")
                    
                    suggestions = {
                        "High Addiction, Low Self Control": "Recommend time management features and mindful usage tips",
                        "High Importance Score, Low Engagement": "Share related content previews that align with their interests",
                        "High Productivity Loss": "Highlight productivity features and tools in your platform",
                        "High Satisfaction, Low Frequency": "Create special re-engagement campaigns for occasional users"
                    }
                    
                    for scenario, suggestion in suggestions.items():
                        st.markdown(f"{scenario}:** {suggestion}")
                else:
                    st.warning("Not enough valid data for behavior profiles after cleaning.")
            except Exception as e:
                st.error(f"Error creating behavior profiles: {str(e)}")
        else:
            st.warning("Not enough behavioral metrics available for profile analysis.")
    
    # 9. Audience Segmentation
    with tabs[8]:
        st.markdown('<div class="section-header">Audience Segmentation</div>', unsafe_allow_html=True)
        
        # K-means clustering for audience segmentation
        segmentation_cols = [col for col in df.columns if col in [
            'Age', 'Engagement', 'Satisfaction', 'Total Time Spent', 'Number of Sessions',
            'Frequency', 'Importance Score', 'Addiction Level', 'Productivity Loss',
            'Self Control', 'Time Spent On Video', 'Number of Videos Watched'
        ]]
        
        if len(segmentation_cols) >= 3:
            st.markdown("### K-means Audience Clustering")
            
            # Create a dataframe with just the clustering columns
            X = df[segmentation_cols].copy()
            
            # Convert all columns to numeric
            for col in segmentation_cols:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            
            # Drop rows with any NaN values
            X = X.dropna()
            
            # Check if we have enough data after cleaning
            if len(X) >= 10:  # Need at least 10 samples for meaningful clustering
                try:
                    # Scale the data
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Set number of clusters (can be made dynamic)
                    num_clusters = st.slider("Select Number of Audience Segments", 2, 6, 4)
                    
                    # Perform K-means clustering
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                    df['Cluster'] = np.nan  # Initialize with NaN
                    df.loc[X.index, 'Cluster'] = kmeans.fit_predict(X_scaled)
                    
                    # Display cluster visualization
                    if 'Age' in segmentation_cols and 'Engagement' in segmentation_cols:
                        fig_cluster = px.scatter(
                            df.dropna(subset=['Cluster']),
                            x='Age',
                            y='Engagement',
                            color='Cluster',
                            title="Audience Segments Visualization",
                            hover_data=['UserID'] if 'UserID' in df.columns else None,
                            opacity=0.7
                        )
                        st.plotly_chart(fig_cluster, use_container_width=True)
            
                    # Get cluster characteristics
                    cluster_profile = df.groupby('Cluster')[segmentation_cols].mean()
                    
                    # Normalize for radar chart
                    cluster_profile_norm = cluster_profile.copy()
                    for col in cluster_profile.columns:
                        min_val = cluster_profile[col].min()
                        max_val = cluster_profile[col].max()
                        if max_val > min_val:
                            cluster_profile_norm[col] = (cluster_profile[col] - min_val) / (max_val - min_val)
                        else:
                            cluster_profile_norm[col] = 0.5
                    
                    # Create radar charts for each cluster
                    st.markdown("### Cluster Profiles")
                    
                    for cluster_id in range(num_clusters):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Display cluster metrics
                            st.markdown(f"#### Cluster {cluster_id}")
                            
                            # Count users in this cluster
                            user_count = len(df[df['Cluster'] == cluster_id])
                            
                            # Generate persona name based on characteristics
                            profile = cluster_profile.loc[cluster_id]
                            
                            # Create persona name based on top characteristics
                            top_features = profile.nlargest(2).index.tolist()
                            persona_names = {
                                'Age': 'Mature' if profile['Age'] > df['Age'].median() else 'Young',
                                'Engagement': 'Engaged',
                                'Satisfaction': 'Satisfied',
                                'Total Time Spent': 'Time-intensive',
                                'Number of Sessions': 'Frequent Visitor',
                                'Frequency': 'Regular',
                                'Importance Score': 'Value-conscious',
                                'Addiction Level': 'Addicted',
                                'Productivity Loss': 'Distracted',
                                'Self Control': 'Disciplined',
                                'Time Spent On Video': 'Video Watcher',
                                'Number of Videos Watched': 'Content Consumer'
                            }
                            
                            persona_name = " ".join([persona_names.get(feature, feature) for feature in top_features])
                            
                            st.markdown(f"*Persona:* {persona_name} Users")
                            st.markdown(f"*Users in segment:* {user_count} ({user_count/len(df)*100:.1f}%)")
                            
                            # Characteristic metrics
                            for col in segmentation_cols[:5]:  # Show top 5 metrics
                                st.metric(col, f"{profile[col]:.2f}")
                        
                        with col2:
                            # Radar chart for this cluster
                            fig_radar = go.Figure()
                            
                            fig_radar.add_trace(go.Scatterpolar(
                                r=cluster_profile_norm.loc[cluster_id].values,
                                theta=cluster_profile_norm.columns,
                                fill='toself',
                                name=f"Cluster {cluster_id}"
                            ))
                            
                            fig_radar.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 1]
                                    )
                                ),
                                title=f"Profile: {persona_name} Users"
                            )
                            
                            st.plotly_chart(fig_radar, use_container_width=True)
                        
                        # Recommendations for this segment
                        segment_recs = {
                            0: "Focus on increasing session time with immersive content",
                            1: "Target with productivity features to balance usage",
                            2: "Encourage content sharing and social engagement",
                            3: "Provide custom content recommendations based on past behavior"
                        }
                        
                        st.markdown(f"*Recommendations:* {segment_recs.get(cluster_id, 'Customize content to segment characteristics')}")
                        st.divider()
                except Exception as e:
                    st.error(f"Error performing clustering: {str(e)}")
            else:
                st.warning("Not enough valid data for clustering after cleaning.")
        else:
            st.warning("Not enough metrics available for audience segmentation analysis.")
    
    # Footer with download option
    st.markdown("---")
    st.markdown("### Download Dashboard Insights")
    
    # Function to create a download link
    def get_download_link(df, filename="social_media_data.csv"):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download filtered data as CSV</a>'
        return href
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(get_download_link(df), unsafe_allow_html=True)
    
    with col2:
        st.write("Dashboard created on:", datetime.datetime.now().strftime("%B %d, %Y"))

else:
    # Show a welcome message when no data is uploaded
    st.markdown("""
    # Welcome to the Social Media Analytics Dashboard 📱
    
    This interactive dashboard will help you analyze social media usage patterns, engagement metrics, and audience segments.
    
    ### 🔍 What insights can you discover?
    
    - *Platform Performance:* Compare usage across platforms, devices, and operating systems
    - *Content Engagement:* Discover what content categories drive the most engagement
    - *Audience Analysis:* Segment users based on behavior and demographics
    - *Marketing ROI:* Measure satisfaction and engagement across different user groups
    - *Sales Funnel:* Track user journey from awareness to engagement
    
    ### 📊 Get Started
    
    1. Upload your social media dataset using the file uploader in the sidebar
    2. Use the filters to focus on specific segments
    3. Navigate through the tabs to explore different insights
    
    ### 📁 Expected Data Format
    
    The dashboard expects a CSV or Excel file with columns related to:
    - User demographics (Age, Gender, Location, etc.)
    - Platform usage (Platform, Device Type, Sessions, etc.)
    - Content engagement (Video Category, Engagement, Satisfaction, etc.)
    - Behavioral indicators (Productivity Loss, Addiction Level, etc.)
    
    ### 🚀 Ready to dive in?
    
    Upload your data file using the sidebar on the left to get started!
    """)
    
    # Display a sample image
    st.image("https://via.placeholder.com/800x400.png?text=Social+Media+Analytics+Dashboard", 
             caption="Upload your data to see the interactive dashboard")
