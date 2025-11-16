import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import traceback

# Machine Learning
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, 
    confusion_matrix, classification_report, 
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import BayesianGaussianMixture
from mlxtend.frequent_patterns import apriori, association_rules

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="📊 Collectibles Analytics Pro",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================

st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        padding: 25px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
    }
    
    /* Success box */
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Info box */
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Warning box */
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 0 24px;
        background-color: white;
        border-radius: 8px;
        color: #31333F;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Download button */
    .download-btn {
        background-color: #667eea;
        color: white;
        padding: 10px 20px;
        text-decoration: none;
        border-radius: 5px;
        display: inline-block;
        margin: 5px;
        font-weight: bold;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #666;
        border-top: 2px solid #eee;
        margin-top: 50px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'cluster_results' not in st.session_state:
    st.session_state.cluster_results = None
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = None
if 'regression_results' not in st.session_state:
    st.session_state.regression_results = None
if 'association_rules' not in st.session_state:
    st.session_state.association_rules = None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_data
def load_default_data():
    """Load default synthetic dataset"""
    try:
        df = pd.read_csv('Collectibles_Platform_Survey_Synthetic_Data.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ Default dataset not found! Please upload a CSV file.")
        return None

def load_uploaded_data(uploaded_file):
    """Load uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def create_download_link(df, filename, link_text):
    """Create a download link for dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-btn">📥 {link_text}</a>'
    return href

def download_figure(fig, filename):
    """Create download link for matplotlib figure"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}" class="download-btn">📥 Download Chart</a>'
    return href

@st.cache_data
def prepare_features(df):
    """Comprehensive feature engineering"""
    try:
        df_processed = df.copy()
        
        # Encode Age
        age_map = {'18-22': 1, '23-27': 2, '28-32': 3, '33-37': 4, '38-45': 5, '46+': 6}
        df_processed['Age_Encoded'] = df_processed['Q1_Age'].map(age_map).fillna(3)
        
        # Encode Income
        income_map = {
            'Below 3 Lakhs': 1, '3-6 Lakhs': 2, '6-10 Lakhs': 3, 
            '10-15 Lakhs': 4, '15-25 Lakhs': 5, '25-50 Lakhs': 6, 'Above 50 Lakhs': 7
        }
        df_processed['Income_Encoded'] = df_processed['Q5_Income'].map(income_map).fillna(3)
        
        # Encode Education
        education_map = {
            'High School': 1, 'Undergraduate': 2, 'Graduate': 3, 
            'Post-Graduate': 4, 'Doctorate': 5
        }
        df_processed['Education_Encoded'] = df_processed['Q6_Education'].map(education_map).fillna(3)
        
        # City Tier
        metro_cities = ['Mumbai', 'Delhi/NCR', 'Bangalore', 'Pune', 'Hyderabad', 'Chennai']
        tier1_cities = ['Kolkata', 'Ahmedabad', 'Other Metro']
        df_processed['City_Tier'] = df_processed['Q3_City'].apply(
            lambda x: 3 if x in metro_cities else (2 if x in tier1_cities else 1)
        )
        
        # Employment Status
        df_processed['Is_Employed'] = df_processed['Q4_Occupation'].str.contains('Employed', na=False).astype(int)
        
        # Behavioral Encoding
        awareness_map = {
            'Not aware at all': 1, 'Heard about it but don\'t know much': 2,
            'Somewhat aware': 3, 'Yes, very aware': 4
        }
        df_processed['Awareness_Encoded'] = df_processed['Q7_Awareness'].map(awareness_map).fillna(2)
        
        purchase_map = {
            'No, not interested': 1, 'No, but interested': 2,
            'Yes, once or twice': 3, 'Yes, multiple times': 4
        }
        df_processed['Purchase_Encoded'] = df_processed['Q8_Purchase'].map(purchase_map).fillna(2)
        
        browse_map = {
            'Never': 1, 'Rarely': 2, 'Once a month': 3,
            'Once a week': 4, '2-3 times a week': 5, 'Daily': 6
        }
        df_processed['Browse_Encoded'] = df_processed['Q10_BrowseFrequency'].map(browse_map).fillna(3)
        
        auth_map = {
            'Not important at all': 1, 'Not very important': 2,
            'Somewhat important': 3, 'Very important': 4, 'Extremely important': 5
        }
        df_processed['Auth_Encoded'] = df_processed['Q13_AuthImportance'].map(auth_map).fillna(3)
        
        # Decision Speed
        speed_map = {
            'Very slow (months)': 1, 'Slow (weeks)': 2, 'Moderate (1-3 days)': 3,
            'Quick (same day)': 4, 'Impulsive (within minutes)': 5
        }
        df_processed['Decision_Speed'] = df_processed['Q27_DecisionSpeed'].map(speed_map).fillna(3)
        
        # Community Importance
        community_map = {
            'Not important': 1, 'Neutral': 2, 'Somewhat important': 3, 'Very important': 4
        }
        df_processed['Community_Encoded'] = df_processed['Q20_Community'].map(community_map).fillna(2)
        
        # Target Variable for Classification (Binary)
        likelihood_map = {
            'Definitely won\'t use': 0, 'Not likely': 0,
            'Somewhat likely': 0, 'Very likely': 1, 'Definitely will use': 1
        }
        df_processed['Target_Binary'] = df_processed['Q34_Likelihood'].map(likelihood_map).fillna(0)
        
        # Loyalty Score (Engineered Feature)
        df_processed['Loyalty_Score'] = (
            (df_processed['Purchase_Encoded'] / 4 * 40) +
            (df_processed['Browse_Encoded'] / 6 * 30) +
            (df_processed['Awareness_Encoded'] / 4 * 30)
        )
        
        # Category Interest Counts
        category_cols = [col for col in df.columns if col.startswith('Q14_') and col != 'Q14_None']
        if category_cols:
            df_processed['Total_Categories'] = df[category_cols].sum(axis=1)
        else:
            df_processed['Total_Categories'] = 0
        
        return df_processed
    
    except Exception as e:
        st.error(f"Error in feature preparation: {str(e)}")
        return df

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Create confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Likely', 'Likely'],
                yticklabels=['Not Likely', 'Likely'],
                cbar_kws={'label': 'Count'})
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    
    # ========================================================================
    # HEADER
    # ========================================================================
    
    st.markdown(
        '<div class="main-header">💎 Collectibles Platform Analytics Dashboard</div>', 
        unsafe_allow_html=True
    )
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    
    st.sidebar.title("⚙️ Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Data Upload Section
    st.sidebar.header("📁 Data Management")
    
    data_source = st.sidebar.radio(
        "Choose Data Source:",
        ["📊 Use Default Dataset", "📤 Upload Custom Dataset"]
    )
    
    if data_source == "📤 Upload Custom Dataset":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV File", 
            type=['csv'],
            help="Upload a CSV file with the same structure as the synthetic dataset"
        )
        
        if uploaded_file is not None:
            df = load_uploaded_data(uploaded_file)
            if df is not None:
                st.session_state.df = df
                st.sidebar.success("✅ Custom data loaded successfully!")
        else:
            if st.session_state.df is None:
                st.sidebar.warning("⚠️ Please upload a CSV file")
                df = None
            else:
                df = st.session_state.df
    else:
        df = load_default_data()
        st.session_state.df = df
        if df is not None:
            st.sidebar.info("ℹ️ Using default synthetic dataset")
    
    if df is None:
        st.error("❌ No data loaded. Please upload a CSV file or ensure the default dataset exists.")
        st.stop()
    
    # Display dataset info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Dataset Info")
    st.sidebar.metric("Total Records", len(df))
    st.sidebar.metric("Total Features", len(df.columns))
    
    # Prepare features
    with st.spinner("🔄 Processing features..."):
        df_processed = prepare_features(df)
        st.session_state.df_processed = df_processed
    
    st.sidebar.success("✅ Features prepared")
    
    # Algorithm Selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Quick Navigation")
    st.sidebar.markdown("""
    - **Home**: Overview & Statistics
    - **Classification**: Predict adoption
    - **Clustering**: Customer segments
    - **Association Rules**: Pattern discovery
    - **Regression**: Pricing predictions
    """)
    
    # ========================================================================
    # MAIN TABS
    # ========================================================================
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Home",
        "🎯 Classification",
        "🔮 Clustering",
        "🔗 Association Rules",
        "💰 Regression & Pricing"
    ])
    
    # ========================================================================
    # TAB 1: HOME
    # ========================================================================
    
    with tab1:
        st.header("📊 Dataset Overview & Quick Insights")
        
        # Key Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📝 Total Respondents", len(df), help="Number of survey responses")
        
        with col2:
            adoption_rate = (df['Q34_Likelihood'].isin(['Definitely will use', 'Very likely']).sum() / len(df) * 100)
            st.metric("📈 Adoption Rate", f"{adoption_rate:.1f}%", help="% willing to use platform")
        
        with col3:
            avg_sneakers = df['Q30_Sneakers_WTP'].mean()
            st.metric("👟 Avg WTP (Sneakers)", f"₹{avg_sneakers:,.0f}", help="Average willingness to pay")
        
        with col4:
            high_income_pct = (df['Q5_Income'].isin(['15-25 Lakhs', '25-50 Lakhs', 'Above 50 Lakhs']).sum() / len(df) * 100)
            st.metric("💰 High Income %", f"{high_income_pct:.1f}%", help="% earning 15L+ annually")
        
        with col5:
            aware_pct = (df['Q7_Awareness'].isin(['Yes, very aware', 'Somewhat aware']).sum() / len(df) * 100)
            st.metric("🧠 Market Awareness", f"{aware_pct:.1f}%", help="% aware of collectibles market")
        
        st.markdown("---")
        
        # Data Preview and Visualizations
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📋 Data Preview")
            st.dataframe(
                df.head(10),
                use_container_width=True,
                height=400
            )
            
            # Download full dataset
            st.markdown(
                create_download_link(df, "full_dataset.csv", "Download Full Dataset"),
                unsafe_allow_html=True
            )
        
        with col2:
            st.subheader("📊 Quick Visualizations")
            
            viz_option = st.selectbox(
                "Select Visualization",
                ["Age Distribution", "Income Distribution", "Adoption Likelihood", "WTP Distribution"]
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if viz_option == "Age Distribution":
                age_counts = df['Q1_Age'].value_counts().sort_index()
                colors = plt.cm.viridis(np.linspace(0, 1, len(age_counts)))
                age_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black', linewidth=1.5)
                ax.set_title('Age Distribution', fontsize=14, fontweight='bold', pad=20)
                ax.set_xlabel('Age Group', fontsize=12, fontweight='bold')
                ax.set_ylabel('Count', fontsize=12, fontweight='bold')
                plt.xticks(rotation=45, ha='right')
            
            elif viz_option == "Income Distribution":
                income_order = ['Below 3 Lakhs', '3-6 Lakhs', '6-10 Lakhs', '10-15 Lakhs', 
                               '15-25 Lakhs', '25-50 Lakhs', 'Above 50 Lakhs']
                income_counts = df['Q5_Income'].value_counts().reindex(income_order)
                colors = plt.cm.plasma(np.linspace(0, 1, len(income_counts)))
                income_counts.plot(kind='barh', ax=ax, color=colors, edgecolor='black', linewidth=1.5)
                ax.set_title('Income Distribution', fontsize=14, fontweight='bold', pad=20)
                ax.set_xlabel('Count', fontsize=12, fontweight='bold')
                ax.set_ylabel('Income Bracket', fontsize=12, fontweight='bold')
            
            elif viz_option == "Adoption Likelihood":
                likelihood_order = ['Definitely won\'t use', 'Not likely', 'Somewhat likely', 
                                   'Very likely', 'Definitely will use']
                likelihood_counts = df['Q34_Likelihood'].value_counts().reindex(likelihood_order)
                colors = ['#d32f2f', '#f57c00', '#fbc02d', '#7cb342', '#388e3c']
                likelihood_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black', linewidth=1.5)
                ax.set_title('Platform Adoption Likelihood', fontsize=14, fontweight='bold', pad=20)
                ax.set_xlabel('Likelihood', fontsize=12, fontweight='bold')
                ax.set_ylabel('Count', fontsize=12, fontweight='bold')
                plt.xticks(rotation=45, ha='right')
            
            else:  # WTP Distribution
                data_to_plot = [df['Q30_Sneakers_WTP'], df['Q30_Watches_WTP'], df['Q30_Cards_WTP']]
                ax.boxplot(data_to_plot, labels=['Sneakers', 'Watches', 'Cards'], patch_artist=True)
                ax.set_title('Willingness to Pay Distribution', fontsize=14, fontweight='bold', pad=20)
                ax.set_ylabel('Price (₹)', fontsize=12, fontweight='bold')
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/1000:.0f}K'))
            
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            st.pyplot(fig)
            
            st.markdown(download_figure(fig, f"{viz_option.replace(' ', '_')}.png"), unsafe_allow_html=True)
        
        # Dataset Statistics
        st.markdown("---")
        st.subheader("📈 Detailed Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🎯 Behavioral Stats**")
            st.write(f"• High Awareness: {(df['Q7_Awareness'] == 'Yes, very aware').sum()}")
            st.write(f"• Past Purchasers: {df['Q8_Purchase'].isin(['Yes, once or twice', 'Yes, multiple times']).sum()}")
            st.write(f"• Daily Browsers: {(df['Q10_BrowseFrequency'] == 'Daily').sum()}")
        
        with col2:
            st.markdown("**💰 Spending Stats**")
            st.write(f"• Avg Sneakers WTP: ₹{df['Q30_Sneakers_WTP'].mean():,.0f}")
            st.write(f"• Avg Watches WTP: ₹{df['Q30_Watches_WTP'].mean():,.0f}")
            st.write(f"• Avg Cards WTP: ₹{df['Q30_Cards_WTP'].mean():,.0f}")
        
        with col3:
            st.markdown("**🏙️ Geographic Stats**")
            top_cities = df['Q3_City'].value_counts().head(3)
            for city, count in top_cities.items():
                st.write(f"• {city}: {count}")
    
    # ========================================================================
    # TAB 2: CLASSIFICATION
    # ========================================================================
    
    with tab2:
        st.header("🎯 Classification: Platform Adoption Prediction")
        
        st.markdown("""
        <div class="info-box">
        <strong>Objective:</strong> Predict whether a customer will adopt the platform based on their characteristics.
        <br><strong>Target Variable:</strong> Platform Adoption (Binary: Likely vs Unlikely)
        </div>
        """, unsafe_allow_html=True)
        
        # Feature Selection
        st.subheader("🔧 Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_size = st.slider(
                "Test Set Size (%)", 
                min_value=10, 
                max_value=40, 
                value=20, 
                step=5,
                help="Percentage of data to use for testing"
            ) / 100
        
        with col2:
            random_state = st.number_input(
                "Random Seed",
                min_value=1,
                max_value=100,
                value=42,
                help="For reproducibility"
            )
        
        with col3:
            cv_folds = st.slider(
                "Cross-Validation Folds",
                min_value=3,
                max_value=10,
                value=5,
                help="Number of folds for cross-validation"
            )
        
        # Feature selection
        st.subheader("📊 Feature Selection")
        
        available_features = [
            'Age_Encoded', 'Income_Encoded', 'Education_Encoded', 'City_Tier', 'Is_Employed',
            'Awareness_Encoded', 'Purchase_Encoded', 'Browse_Encoded', 'Auth_Encoded',
            'Decision_Speed', 'Community_Encoded', 'Loyalty_Score',
            'Q30_Sneakers_WTP', 'Q30_Watches_WTP', 'Q30_Cards_WTP', 'Total_Categories'
        ]
        
        selected_features = st.multiselect(
            "Select Features for Training",
            available_features,
            default=['Income_Encoded', 'Purchase_Encoded', 'Auth_Encoded', 
                    'Q30_Sneakers_WTP', 'Loyalty_Score'],
            help="Choose features to train the classification models"
        )
        
        if len(selected_features) < 2:
            st.warning("⚠️ Please select at least 2 features to train models.")
            st.stop()
        
        # Prepare data
        X = df_processed[selected_features].fillna(0)
        y = df_processed['Target_Binary'].fillna(0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", int(len(X) * (1 - test_size)))
        with col2:
            st.metric("Test Samples", int(len(X) * test_size))
        with col3:
            st.metric("Features Used", len(selected_features))
        
        # Model Selection
        st.subheader("🤖 Model Selection")
        
        model_options = st.multiselect(
            "Select Models to Train",
            [
                'Logistic Regression',
                'Random Forest',
                'Gradient Boosting',
                'Decision Tree',
                'SVM',
                'Naive Bayes'
            ],
            default=['Logistic Regression', 'Random Forest', 'Gradient Boosting']
        )
        
        if len(model_options) == 0:
            st.warning("⚠️ Please select at least one model.")
            st.stop()
        
        # Train Models Button
        if st.button("🚀 Train Classification Models", type="primary", use_container_width=True):
            
            with st.spinner("🔄 Training models... This may take a moment..."):
                
                try:
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, stratify=y
                    )
                    
                    # Initialize models
                    models = {}
                    if 'Logistic Regression' in model_options:
                        models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=random_state)
                    if 'Random Forest' in model_options:
                        models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=random_state)
                    if 'Gradient Boosting' in model_options:
                        models['Gradient Boosting'] = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
                    if 'Decision Tree' in model_options:
                        models['Decision Tree'] = DecisionTreeClassifier(max_depth=10, random_state=random_state)
                    if 'SVM' in model_options:
                        models['SVM'] = SVC(kernel='rbf', random_state=random_state)
                    if 'Naive Bayes' in model_options:
                        models['Naive Bayes'] = GaussianNB()
                    
                    results = {}
                    progress_bar = st.progress(0)
                    
                    for idx, (name, model) in enumerate(models.items()):
                        # Train model
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, zero_division=0)
                        recall = recall_score(y_test, y_pred, zero_division=0)
                        f1 = f1_score(y_test, y_pred, zero_division=0)
                        
                        # Cross-validation
                        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
                        
                        results[name] = {
                            'Accuracy': accuracy,
                            'Precision': precision,
                            'Recall': recall,
                            'F1-Score': f1,
                            'CV_Mean': cv_scores.mean(),
                            'CV_Std': cv_scores.std(),
                            'Predictions': y_pred,
                            'Model': model
                        }
                        
                        progress_bar.progress((idx + 1) / len(models))
                    
                    progress_bar.empty()
                    st.session_state.classification_results = {
                        'results': results,
                        'X_test': X_test,
                        'y_test': y_test,
                        'features': selected_features
                    }
                    
                    st.success("✅ All models trained successfully!")
                    
                    # Display Results
                    st.markdown("---")
                    st.subheader("📊 Model Performance Comparison")
                    
                    # Results Table
                    results_df = pd.DataFrame({
                        'Model': list(results.keys()),
                        'Accuracy': [f"{results[m]['Accuracy']:.2%}" for m in results],
                        'Precision': [f"{results[m]['Precision']:.2%}" for m in results],
                        'Recall': [f"{results[m]['Recall']:.2%}" for m in results],
                        'F1-Score': [f"{results[m]['F1-Score']:.2%}" for m in results],
                        'CV Mean': [f"{results[m]['CV_Mean']:.2%}" for m in results],
                        'CV Std': [f"{results[m]['CV_Std']:.3f}" for m in results]
                    })
                    
                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        height=300
                    )
                    
                    # Best Model
                    best_model = max(results.keys(), key=lambda k: results[k]['Accuracy'])
                    
                    st.markdown(f"""
                    <div class="success-box">
                    <h3>🏆 Best Model: {best_model}</h3>
                    <ul>
                        <li><strong>Accuracy:</strong> {results[best_model]['Accuracy']:.2%}</li>
                        <li><strong>Precision:</strong> {results[best_model]['Precision']:.2%}</li>
                        <li><strong>Recall:</strong> {results[best_model]['Recall']:.2%}</li>
                        <li><strong>F1-Score:</strong> {results[best_model]['F1-Score']:.2%}</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Visualizations
                    st.markdown("---")
                    st.subheader("📈 Visualizations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Model Comparison Chart
                        fig1, ax1 = plt.subplots(figsize=(10, 6))
                        
                        metrics_df = pd.DataFrame({
                            'Accuracy': [results[m]['Accuracy'] for m in results],
                            'Precision': [results[m]['Precision'] for m in results],
                            'Recall': [results[m]['Recall'] for m in results],
                            'F1-Score': [results[m]['F1-Score'] for m in results]
                        }, index=list(results.keys()))
                        
                        metrics_df.plot(kind='bar', ax=ax1, rot=45, width=0.8)
                        ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
                        ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
                        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
                        ax1.legend(loc='lower right', fontsize=10)
                        ax1.grid(axis='y', alpha=0.3, linestyle='--')
                        ax1.set_ylim(0, 1)
                        plt.tight_layout()
                        
                        st.pyplot(fig1)
                        st.markdown(download_figure(fig1, "model_comparison.png"), unsafe_allow_html=True)
                    
                    with col2:
                        # Confusion Matrix
                        fig2 = plot_confusion_matrix(
                            y_test, 
                            results[best_model]['Predictions'],
                            title=f"Confusion Matrix - {best_model}"
                        )
                        
                        st.pyplot(fig2)
                        st.markdown(download_figure(fig2, "confusion_matrix.png"), unsafe_allow_html=True)
                    
                    # Feature Importance (if available)
                    if hasattr(results[best_model]['Model'], 'feature_importances_'):
                        st.markdown("---")
                        st.subheader("🔍 Feature Importance Analysis")
                        
                        importance_df = pd.DataFrame({
                            'Feature': selected_features,
                            'Importance': results[best_model]['Model'].feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig3, ax3 = plt.subplots(figsize=(10, 6))
                        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
                        ax3.barh(importance_df['Feature'], importance_df['Importance'], color=colors, edgecolor='black')
                        ax3.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
                        ax3.set_title(f'Feature Importance - {best_model}', fontsize=14, fontweight='bold', pad=20)
                        ax3.grid(axis='x', alpha=0.3, linestyle='--')
                        plt.tight_layout()
                        
                        st.pyplot(fig3)
                        st.markdown(download_figure(fig3, "feature_importance.png"), unsafe_allow_html=True)
                    
                    # Download Predictions
                    st.markdown("---")
                    st.subheader("💾 Download Results")
                    
                    pred_df = pd.DataFrame({
                        'Actual': y_test.values,
                        'Predicted': results[best_model]['Predictions'],
                        'Correct': (y_test.values == results[best_model]['Predictions']).astype(int)
                    })
                    
                    pred_df.index = X_test.index
                    
                    st.markdown(
                        create_download_link(pred_df, "classification_predictions.csv", "Download Predictions"),
                        unsafe_allow_html=True
                    )
                    
                    st.markdown(
                        create_download_link(results_df, "model_performance.csv", "Download Model Metrics"),
                        unsafe_allow_html=True
                    )
                
                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")
                    st.code(traceback.format_exc())
        
        # Display cached results
        elif st.session_state.classification_results is not None:
            st.info("ℹ️ Showing previous results. Click 'Train Classification Models' to retrain.")
    
    # ========================================================================
    # TAB 3: CLUSTERING
    # ========================================================================
    
    with tab3:
        st.header("🔮 Customer Segmentation: Clustering Analysis")
        
        st.markdown("""
        <div class="info-box">
        <strong>Objective:</strong> Segment customers into distinct groups based on their characteristics and behaviors.
        <br><strong>Method:</strong> K-Means Clustering with interactive parameters.
        </div>
        """, unsafe_allow_html=True)
        
        # Clustering Parameters
        st.subheader("⚙️ Clustering Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_clusters = st.slider(
                "Number of Clusters (K)",
                min_value=2,
                max_value=10,
                value=5,
                step=1,
                help="Number of customer segments to create",
                key="n_clusters_slider"
            )
        
        with col2:
            random_state_cluster = st.number_input(
                "Random Seed ",
                min_value=1,
                max_value=100,
                value=42,
                key="cluster_seed"
            )
        
        with col3:
            n_init = st.slider(
                "Number of Initializations",
                min_value=10,
                max_value=50,
                value=20,
                step=5,
                help="Number of times K-Means will run with different centroid seeds"
            )
        
        # Feature Selection
        st.subheader("📊 Feature Selection for Clustering")
        
        clustering_features_available = [
            'Age_Encoded', 'Income_Encoded', 'Education_Encoded', 'City_Tier',
            'Awareness_Encoded', 'Purchase_Encoded', 'Browse_Encoded', 'Auth_Encoded',
            'Decision_Speed', 'Community_Encoded', 'Loyalty_Score',
            'Q30_Sneakers_WTP', 'Q30_Watches_WTP', 'Q30_Cards_WTP', 'Total_Categories'
        ]
        
        clustering_features = st.multiselect(
            "Select Features for Clustering",
            clustering_features_available,
            default=['Income_Encoded', 'Loyalty_Score', 'Q30_Sneakers_WTP', 
                    'Q30_Watches_WTP', 'Purchase_Encoded'],
            help="Choose features that define customer segments"
        )
        
        if len(clustering_features) < 2:
            st.warning("⚠️ Please select at least 2 features for clustering.")
            st.stop()
        
        # Prepare data
        X_cluster = df_processed[clustering_features].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        st.metric("Features Selected", len(clustering_features))
        
        # Run Clustering
        if st.button("🚀 Run K-Means Clustering", type="primary", use_container_width=True):
            
            with st.spinner("🔄 Running K-Means clustering..."):
                
                try:
                    # K-Means
                    kmeans = KMeans(
                        n_clusters=n_clusters,
                        random_state=random_state_cluster,
                        n_init=n_init,
                        max_iter=500
                    )
                    clusters = kmeans.fit_predict(X_scaled)
                    
                    df_processed['Cluster'] = clusters
                    
                    # Calculate metrics
                    silhouette = silhouette_score(X_scaled, clusters)
                    davies_bouldin = davies_bouldin_score(X_scaled, clusters)
                    inertia = kmeans.inertia_
                    
                    st.session_state.cluster_results = {
                        'kmeans': kmeans,
                        'clusters': clusters,
                        'silhouette': silhouette,
                        'davies_bouldin': davies_bouldin,
                        'inertia': inertia,
                        'X_scaled': X_scaled,
                        'scaler': scaler,
                        'features': clustering_features,
                        'n_clusters': n_clusters
                    }
                    
                    st.success(f"✅ K-Means completed! Created {n_clusters} customer segments.")
                    
                    # Display Metrics
                    st.markdown("---")
                    st.subheader("📊 Clustering Quality Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Silhouette Score",
                            f"{silhouette:.3f}",
                            help="Range: -1 to 1. Higher is better (>0.5 is good)"
                        )
                    
                    with col2:
                        st.metric(
                            "Davies-Bouldin Index",
                            f"{davies_bouldin:.3f}",
                            help="Lower is better (closer to 0)"
                        )
                    
                    with col3:
                        st.metric(
                            "Inertia",
                            f"{inertia:,.0f}",
                            help="Within-cluster sum of squares (lower is better)"
                        )
                    
                    # Cluster Distribution
                    st.markdown("---")
                    st.subheader("📊 Cluster Size Distribution")
                    
                    cluster_counts = pd.Series(clusters).value_counts().sort_index()
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        for cluster_id, count in cluster_counts.items():
                            percentage = (count / len(clusters)) * 100
                            st.metric(
                                f"Cluster {cluster_id}",
                                f"{count} customers",
                                f"{percentage:.1f}%"
                            )
                    
                    with col2:
                        fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
                        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
                        ax_dist.bar(cluster_counts.index, cluster_counts.values, color=colors, edgecolor='black', linewidth=1.5)
                        ax_dist.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
                        ax_dist.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
                        ax_dist.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold', pad=20)
                        ax_dist.grid(axis='y', alpha=0.3, linestyle='--')
                        
                        for i, count in enumerate(cluster_counts.values):
                            ax_dist.text(i, count + 5, str(count), ha='center', fontweight='bold')
                        
                        st.pyplot(fig_dist)
                        st.markdown(download_figure(fig_dist, "cluster_distribution.png"), unsafe_allow_html=True)
                    
                    # Interactive Persona Explorer
                    st.markdown("---")
                    st.subheader("👥 Interactive Cluster Profile Explorer")
                    
                    selected_cluster = st.selectbox(
                        "Select Cluster to Explore",
                        range(n_clusters),
                        format_func=lambda x: f"Cluster {x} ({cluster_counts[x]} customers)"
                    )
                    
                    cluster_data = df_processed[df_processed['Cluster'] == selected_cluster]
                    
                    st.markdown(f"""
                    <div class="metric-card">
                    <h3>Cluster {selected_cluster} Profile</h3>
                    <p><strong>Size:</strong> {len(cluster_data)} customers ({len(cluster_data)/len(df_processed)*100:.1f}% of total)</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Profile Details
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**👤 Demographics**")
                        st.write(f"• **Age:** {cluster_data['Q1_Age'].mode()[0]}")
                        st.write(f"• **Income:** {cluster_data['Q5_Income'].mode()[0]}")
                        st.write(f"• **Education:** {cluster_data['Q6_Education'].mode()[0]}")
                        st.write(f"• **City:** {cluster_data['Q3_City'].mode()[0]}")
                    
                    with col2:
                        st.markdown("**🎭 Behavior**")
                        st.write(f"• **Awareness:** {cluster_data['Q7_Awareness'].mode()[0]}")
                        st.write(f"• **Purchase:** {cluster_data['Q8_Purchase'].mode()[0]}")
                        st.write(f"• **Browse:** {cluster_data['Q10_BrowseFrequency'].mode()[0]}")
                        st.write(f"• **Loyalty:** {cluster_data['Loyalty_Score'].mean():.1f}/100")
                    
                    with col3:
                        st.markdown("**💰 Spending**")
                        st.write(f"• **Sneakers:** ₹{cluster_data['Q30_Sneakers_WTP'].mean():,.0f}")
                        st.write(f"• **Watches:** ₹{cluster_data['Q30_Watches_WTP'].mean():,.0f}")
                        st.write(f"• **Cards:** ₹{cluster_data['Q30_Cards_WTP'].mean():,.0f}")
                        adoption = (cluster_data['Target_Binary'].sum() / len(cluster_data)) * 100
                        st.write(f"• **Adoption:** {adoption:.1f}%")
                    
                    # Complete Persona Table
                    st.markdown("---")
                    st.subheader("📋 Complete Persona Summary Table")
                    
                    persona_data = []
                    for cid in range(n_clusters):
                        cdata = df_processed[df_processed['Cluster'] == cid]
                        adoption_rate = (cdata['Target_Binary'].sum() / len(cdata)) * 100
                        
                        persona_data.append({
                            'Cluster': cid,
                            'Size': len(cdata),
                            '%': f"{len(cdata)/len(df_processed)*100:.1f}%",
                            'Age': cdata['Q1_Age'].mode()[0],
                            'Income': cdata['Q5_Income'].mode()[0],
                            'Loyalty': f"{cdata['Loyalty_Score'].mean():.0f}",
                            'Sneakers WTP': f"₹{cdata['Q30_Sneakers_WTP'].mean():,.0f}",
                            'Watches WTP': f"₹{cdata['Q30_Watches_WTP'].mean():,.0f}",
                            'Cards WTP': f"₹{cdata['Q30_Cards_WTP'].mean():,.0f}",
                            'Adoption %': f"{adoption_rate:.1f}%"
                        })
                    
                    persona_df = pd.DataFrame(persona_data)
                    st.dataframe(persona_df, use_container_width=True, height=300)
                    
                    st.markdown(
                        create_download_link(persona_df, "persona_summary.csv", "Download Persona Table"),
                        unsafe_allow_html=True
                    )
                    
                    # PCA Visualization
                    st.markdown("---")
                    st.subheader("🎨 Cluster Visualization (PCA)")
                    
                    pca = PCA(n_components=2, random_state=random_state_cluster)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    fig_pca, ax_pca = plt.subplots(figsize=(14, 10))
                    
                    scatter = ax_pca.scatter(
                        X_pca[:, 0], X_pca[:, 1],
                        c=clusters,
                        cmap='Set3',
                        s=100,
                        alpha=0.6,
                        edgecolors='black',
                        linewidth=0.5
                    )
                    
                    # Plot centroids
                    centroids_pca = pca.transform(kmeans.cluster_centers_)
                    ax_pca.scatter(
                        centroids_pca[:, 0], centroids_pca[:, 1],
                        c='red',
                        marker='X',
                        s=500,
                        edgecolors='black',
                        linewidth=2,
                        label='Centroids',
                        zorder=5
                    )
                    
                    ax_pca.set_xlabel(
                        f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)',
                        fontsize=12,
                        fontweight='bold'
                    )
                    ax_pca.set_ylabel(
                        f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)',
                        fontsize=12,
                        fontweight='bold'
                    )
                    ax_pca.set_title(
                        f'Customer Clusters Visualization (K={n_clusters})',
                        fontsize=14,
                        fontweight='bold',
                        pad=20
                    )
                    ax_pca.legend(fontsize=11)
                    ax_pca.grid(True, alpha=0.3, linestyle='--')
                    plt.colorbar(scatter, ax=ax_pca, label='Cluster ID')
                    
                    st.pyplot(fig_pca)
                    st.markdown(download_figure(fig_pca, "cluster_pca_visualization.png"), unsafe_allow_html=True)
                    
                    # Cluster Characteristics Heatmap
                    st.markdown("---")
                    st.subheader("🔥 Cluster Characteristics Heatmap")
                    
                    # Calculate means for each cluster
                    cluster_means = df_processed.groupby('Cluster')[clustering_features].mean()
                    
                    # Normalize for better visualization
                    from sklearn.preprocessing import MinMaxScaler
                    scaler_viz = MinMaxScaler()
                    cluster_means_norm = pd.DataFrame(
                        scaler_viz.fit_transform(cluster_means.T).T,
                        columns=cluster_means.columns,
                        index=cluster_means.index
                    )
                    
                    fig_heatmap, ax_heatmap = plt.subplots(figsize=(14, 8))
                    sns.heatmap(
                        cluster_means_norm,
                        annot=True,
                        fmt='.2f',
                        cmap='RdYlGn',
                        cbar_kws={'label': 'Normalized Value (0-1)'},
                        linewidths=2,
                        linecolor='white',
                        ax=ax_heatmap,
                        vmin=0,
                        vmax=1
                    )
                    ax_heatmap.set_xlabel('Features', fontsize=12, fontweight='bold')
                    ax_heatmap.set_ylabel('Cluster ID', fontsize=12, fontweight='bold')
                    ax_heatmap.set_title(
                        'Cluster Characteristics Heatmap\n(Normalized: Green=High, Red=Low)',
                        fontsize=14,
                        fontweight='bold',
                        pad=20
                    )
                    ax_heatmap.set_yticklabels([f'Cluster {i}' for i in range(n_clusters)], rotation=0)
                    plt.tight_layout()
                    
                    st.pyplot(fig_heatmap)
                    st.markdown(download_figure(fig_heatmap, "cluster_heatmap.png"), unsafe_allow_html=True)
                    
                    # Download Results
                    st.markdown("---")
                    st.subheader("💾 Download Clustering Results")
                    
                    cluster_assignments = df_processed[['ResponseID', 'Cluster']].copy()
                    cluster_assignments['Cluster_Name'] = cluster_assignments['Cluster'].apply(
                        lambda x: f"Cluster_{x}"
                    )
                    
                    st.markdown(
                        create_download_link(cluster_assignments, "cluster_assignments.csv", "Download Cluster Assignments"),
                        unsafe_allow_html=True
                    )
                    
                    st.markdown(
                        create_download_link(cluster_means, "cluster_profiles.csv", "Download Cluster Profiles"),
                        unsafe_allow_html=True
                    )
                
                except Exception as e:
                    st.error(f"❌ Error during clustering: {str(e)}")
                    st.code(traceback.format_exc())
    
    # ========================================================================
    # TAB 4: ASSOCIATION RULES
    # ========================================================================
    
    with tab4:
        st.header("🔗 Association Rule Mining: Pattern Discovery")
        
        st.markdown("""
        <div class="info-box">
        <strong>Objective:</strong> Discover interesting associations and patterns between different product categories and customer preferences.
        <br><strong>Method:</strong> Apriori Algorithm with adjustable thresholds.
        </div>
        """, unsafe_allow_html=True)
        
        # Parameters
        st.subheader("⚙️ Mining Parameters")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            min_support = st.slider(
                "Minimum Support (%)",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
                help="Minimum frequency of itemset occurrence"
            ) / 100
        
        with col2:
            min_confidence = st.slider(
                "Minimum Confidence (%)",
                min_value=50,
                max_value=95,
                value=70,
                step=5,
                help="Minimum probability of rule being true"
            ) / 100
        
        with col3:
            min_lift = st.slider(
                "Minimum Lift",
                min_value=1.0,
                max_value=3.0,
                value=1.2,
                step=0.1,
                help="Minimum lift value (>1 means positive association)"
            )
        
        with col4:
            top_n_rules = st.slider(
                "Top N Rules",
                min_value=5,
                max_value=30,
                value=10,
                step=5,
                help="Number of top rules to display"
            )
        
        # Prepare transaction data
        st.subheader("📊 Transaction Data Preparation")
        
        # Select columns for association analysis
        category_cols = [col for col in df.columns if col.startswith('Q14_') and 'None' not in col]
        sneaker_cols = [col for col in df.columns if col.startswith('Q15_') and 'Not_interested' not in col]
        watch_cols = [col for col in df.columns if col.startswith('Q16_') and 'Not_interested' not in col]
        card_cols = [col for col in df.columns if col.startswith('Q17_') and 'Not_interested' not in col]
        
        all_transaction_cols = category_cols + sneaker_cols + watch_cols + card_cols
        
        if len(all_transaction_cols) == 0:
            st.warning("⚠️ No transaction columns found in the dataset.")
            st.stop()
        
        # Clean column names
        def clean_col_name(col):
            col = col.replace('Q14_', '[CAT] ')
            col = col.replace('Q15_', '[SNEAKER] ')
            col = col.replace('Q16_', '[WATCH] ')
            col = col.replace('Q17_', '[CARD] ')
            col = col.replace('_', ' ')
            return col
        
        transaction_df = df[all_transaction_cols].astype(bool).copy()
        transaction_df.columns = [clean_col_name(col) for col in transaction_df.columns]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transactions", len(transaction_df))
        with col2:
            st.metric("Total Items", len(transaction_df.columns))
        with col3:
            avg_items = transaction_df.sum(axis=1).mean()
            st.metric("Avg Items per Transaction", f"{avg_items:.1f}")
        
        # Run Apriori
        if st.button("🚀 Generate Association Rules", type="primary", use_container_width=True):
            
            with st.spinner("🔄 Running Apriori algorithm... This may take a moment..."):
                
                try:
                    # Find frequent itemsets
                    frequent_itemsets = apriori(
                        transaction_df,
                        min_support=min_support,
                        use_colnames=True
                    )
                    
                    if len(frequent_itemsets) == 0:
                        st.warning(f"""
                        ⚠️ No frequent itemsets found with minimum support of {min_support:.0%}.
                        
                        **Suggestions:**
                        - Try lowering the minimum support threshold
                        - Check if your data has enough transactions
                        """)
                        st.stop()
                    
                    # Generate rules
                    rules = association_rules(
                        frequent_itemsets,
                        metric="confidence",
                        min_threshold=min_confidence
                    )
                    
                    if len(rules) == 0:
                        st.warning(f"""
                        ⚠️ No association rules found with minimum confidence of {min_confidence:.0%}.
                        
                        **Suggestions:**
                        - Try lowering the minimum confidence threshold
                        - Reduce the minimum support threshold
                        """)
                        st.stop()
                    
                    # Filter by lift
                    rules = rules[rules['lift'] >= min_lift]
                    
                    if len(rules) == 0:
                        st.warning(f"⚠️ No rules found with minimum lift of {min_lift:.1f}.")
                        st.stop()
                    
                    # Sort by lift
                    rules = rules.sort_values('lift', ascending=False)
                    
                    st.session_state.association_rules = {
                        'rules': rules,
                        'frequent_itemsets': frequent_itemsets,
                        'parameters': {
                            'support': min_support,
                            'confidence': min_confidence,
                            'lift': min_lift
                        }
                    }
                    
                    st.success(f"✅ Found {len(rules)} association rules!")
                    
                    # Display Metrics
                    st.markdown("---")
                    st.subheader("📊 Mining Results Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Frequent Itemsets", len(frequent_itemsets))
                    with col2:
                        st.metric("Total Rules", len(rules))
                    with col3:
                        st.metric("Avg Confidence", f"{rules['confidence'].mean():.1%}")
                    with col4:
                        st.metric("Avg Lift", f"{rules['lift'].mean():.2f}x")
                    
                    # Top Rules
                    st.markdown("---")
                    st.subheader(f"🏆 Top {min(top_n_rules, len(rules))} Association Rules")
                    
                    top_rules = rules.head(top_n_rules).copy()
                    
                    # Format for display
                    top_rules['Antecedents'] = top_rules['antecedents'].apply(
                        lambda x: ', '.join(list(x))
                    )
                    top_rules['Consequents'] = top_rules['consequents'].apply(
                        lambda x: ', '.join(list(x))
                    )
                    
                    display_rules = top_rules[[
                        'Antecedents', 'Consequents',
                        'support', 'confidence', 'lift', 'conviction'
                    ]].copy()
                    
                    display_rules.columns = [
                        'IF (Antecedents)', 'THEN (Consequents)',
                        'Support', 'Confidence', 'Lift', 'Conviction'
                    ]
                    
                    # Format percentages
                    display_rules_formatted = display_rules.copy()
                    display_rules_formatted['Support'] = display_rules_formatted['Support'].apply(lambda x: f"{x:.2%}")
                    display_rules_formatted['Confidence'] = display_rules_formatted['Confidence'].apply(lambda x: f"{x:.2%}")
                    display_rules_formatted['Lift'] = display_rules_formatted['Lift'].apply(lambda x: f"{x:.2f}x")
                    display_rules_formatted['Conviction'] = display_rules_formatted['Conviction'].apply(lambda x: f"{x:.2f}")
                    
                    st.dataframe(
                        display_rules_formatted,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Detailed Rule Interpretations
                    st.markdown("---")
                    st.subheader("📖 Detailed Rule Interpretations")
                    
                    for idx, (i, rule) in enumerate(top_rules.head(5).iterrows(), 1):
                        with st.expander(f"📌 Rule #{idx}: {rule['Antecedents']} → {rule['Consequents']}"):
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Rule Components:**")
                                st.write(f"• **IF:** {rule['Antecedents']}")
                                st.write(f"• **THEN:** {rule['Consequents']}")
                            
                            with col2:
                                st.markdown("**Metrics:**")
                                st.write(f"• **Support:** {rule['support']:.2%}")
                                st.write(f"• **Confidence:** {rule['confidence']:.2%}")
                                st.write(f"• **Lift:** {rule['lift']:.2f}x")
                                st.write(f"• **Conviction:** {rule['conviction']:.2f}")
                            
                            st.markdown("**📊 Interpretation:**")
                            
                            num_transactions = int(rule['support'] * len(transaction_df))
                            
                            st.info(f"""
                            **What this means:**
                            
                            1. **Support ({rule['support']:.1%})**: This pattern appears in {num_transactions} out of {len(transaction_df)} transactions.
                            
                            2. **Confidence ({rule['confidence']:.0%})**: {rule['confidence']:.0%} of customers interested in **{rule['Antecedents']}** 
                               are also interested in **{rule['Consequents']}**.
                            
                            3. **Lift ({rule['lift']:.2f}x)**: Customers interested in **{rule['Antecedents']}** are {rule['lift']:.1f} times 
                               more likely to be interested in **{rule['Consequents']}** compared to random chance.
                            
                            **Business Action:** Consider bundling or cross-promoting these items together!
                            """)
                    
                    # Visualizations
                    st.markdown("---")
                    st.subheader("📈 Visualizations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Support vs Confidence scatter
                        fig1, ax1 = plt.subplots(figsize=(10, 8))
                        
                        scatter = ax1.scatter(
                            rules['support'],
                            rules['confidence'],
                            s=rules['lift'] * 100,
                            c=rules['lift'],
                            cmap='viridis',
                            alpha=0.6,
                            edgecolors='black',
                            linewidth=1
                        )
                        
                        # Highlight top rules
                        top_indices = rules.head(5).index
                        ax1.scatter(
                            rules.loc[top_indices, 'support'],
                            rules.loc[top_indices, 'confidence'],
                            s=rules.loc[top_indices, 'lift'] * 100,
                            c='red',
                            marker='*',
                            edgecolors='darkred',
                            linewidth=2,
                            label='Top 5 Rules',
                            zorder=5
                        )
                        
                        ax1.axhline(y=min_confidence, color='r', linestyle='--', linewidth=2, alpha=0.5,
                                   label=f'Min Confidence ({min_confidence:.0%})')
                        ax1.axvline(x=min_support, color='b', linestyle='--', linewidth=2, alpha=0.5,
                                   label=f'Min Support ({min_support:.0%})')
                        
                        ax1.set_xlabel('Support', fontsize=12, fontweight='bold')
                        ax1.set_ylabel('Confidence', fontsize=12, fontweight='bold')
                        ax1.set_title('Association Rules: Support vs Confidence\n(Bubble size = Lift)',
                                     fontsize=14, fontweight='bold', pad=20)
                        ax1.legend(loc='lower right', fontsize=10)
                        ax1.grid(True, alpha=0.3, linestyle='--')
                        
                        plt.colorbar(scatter, ax=ax1, label='Lift')
                        
                        st.pyplot(fig1)
                        st.markdown(download_figure(fig1, "rules_scatter.png"), unsafe_allow_html=True)
                    
                    with col2:
                        # Top rules bar chart
                        fig2, ax2 = plt.subplots(figsize=(10, 8))
                        
                        top_10_rules = rules.head(10)
                        rule_labels = [f"Rule {i+1}" for i in range(len(top_10_rules))]
                        
                        x = np.arange(len(rule_labels))
                        width = 0.35
                        
                        bars1 = ax2.barh(x - width/2, top_10_rules['confidence'], width,
                                        label='Confidence', color='skyblue', edgecolor='black')
                        bars2 = ax2.barh(x + width/2, top_10_rules['lift']/5, width,
                                        label='Lift (scaled)', color='lightcoral', edgecolor='black')
                        
                        ax2.set_yticks(x)
                        ax2.set_yticklabels(rule_labels)
                        ax2.set_xlabel('Score', fontsize=12, fontweight='bold')
                        ax2.set_title('Top 10 Rules: Confidence & Lift',
                                     fontsize=14, fontweight='bold', pad=20)
                        ax2.legend(fontsize=10)
                        ax2.grid(axis='x', alpha=0.3, linestyle='--')
                        
                        st.pyplot(fig2)
                        st.markdown(download_figure(fig2, "top_rules_comparison.png"), unsafe_allow_html=True)
                    
                    # Network Graph (if small number of rules)
                    if len(top_rules) <= 15:
                        st.markdown("---")
                        st.subheader("🕸️ Rule Network Visualization")
                        
                        try:
                            import networkx as nx
                            
                            fig3, ax3 = plt.subplots(figsize=(14, 10))
                            
                            G = nx.DiGraph()
                            
                            for idx, rule in top_rules.iterrows():
                                for ant in rule['antecedents']:
                                    for cons in rule['consequents']:
                                        G.add_edge(
                                            ant, cons,
                                            weight=rule['lift'],
                                            confidence=rule['confidence']
                                        )
                            
                            pos = nx.spring_layout(G, k=2, iterations=50)
                            
                            nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue',
                                                  alpha=0.9, edgecolors='black', linewidths=2, ax=ax3)
                            
                            edges = G.edges()
                            weights = [G[u][v]['weight'] for u, v in edges]
                            
                            nx.draw_networkx_edges(G, pos, width=[w*2 for w in weights],
                                                  alpha=0.6, edge_color='gray',
                                                  arrows=True, arrowsize=20, ax=ax3)
                            
                            nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax3)
                            
                            ax3.set_title('Association Rules Network\n(Edge width = Lift strength)',
                                        fontsize=14, fontweight='bold', pad=20)
                            ax3.axis('off')
                            
                            st.pyplot(fig3)
                            st.markdown(download_figure(fig3, "rules_network.png"), unsafe_allow_html=True)
                        
                        except ImportError:
                            st.info("ℹ️ Install networkx for network visualization: `pip install networkx`")
                    
                    # Download Results
                    st.markdown("---")
                    st.subheader("💾 Download Results")
                    
                    # Export rules
                    export_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'conviction']].copy()
                    export_rules['antecedents'] = export_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                    export_rules['consequents'] = export_rules['consequents'].apply(lambda x: ', '.join(list(x)))
                    
                    st.markdown(
                        create_download_link(export_rules, "association_rules.csv", "Download All Rules"),
                        unsafe_allow_html=True
                    )
                    
                    st.markdown(
                        create_download_link(display_rules_formatted, f"top_{top_n_rules}_rules.csv", f"Download Top {top_n_rules} Rules"),
                        unsafe_allow_html=True
                    )
                
                except Exception as e:
                    st.error(f"❌ Error during association rule mining: {str(e)}")
                    st.code(traceback.format_exc())
    
    # ========================================================================
    # TAB 5: REGRESSION & DYNAMIC PRICING
    # ========================================================================
    
    with tab5:
        st.header("💰 Regression Analysis & Dynamic Pricing Engine")
        
        st.markdown("""
        <div class="info-box">
        <strong>Objective:</strong> Predict customer willingness to pay (WTP) and generate personalized pricing recommendations.
        <br><strong>Method:</strong> Multiple regression models with dynamic pricing adjustments.
        </div>
        """, unsafe_allow_html=True)
        
        # Category Selection
        st.subheader("🎯 Select Product Category")
        
        category = st.selectbox(
            "Choose Category for Analysis",
            ['👟 Sneakers', '⌚ Watches', '🃏 Trading Cards'],
            format_func=lambda x: x
        )
        
        category_mapping = {
            '👟 Sneakers': 'Q30_Sneakers_WTP',
            '⌚ Watches': 'Q30_Watches_WTP',
            '🃏 Trading Cards': 'Q30_Cards_WTP'
        }
        
        target_col = category_mapping[category]
        category_name = category.split(' ')[1]
        
        # Configuration
        st.subheader("⚙️ Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_size_reg = st.slider(
                "Test Set Size (%) ",
                min_value=10,
                max_value=40,
                value=20,
                step=5,
                key="reg_test_size"
            ) / 100
        
        with col2:
            random_state_reg = st.number_input(
                "Random Seed  ",
                min_value=1,
                max_value=100,
                value=42,
                key="reg_seed"
            )
        
        with col3:
            cv_folds_reg = st.slider(
                "CV Folds",
                min_value=3,
                max_value=10,
                value=5,
                key="reg_cv"
            )
        
        # Feature Selection
        st.subheader("📊 Feature Selection")
        
        regression_features_available = [
            'Age_Encoded', 'Income_Encoded', 'Education_Encoded', 'City_Tier', 'Is_Employed',
            'Awareness_Encoded', 'Purchase_Encoded', 'Browse_Encoded', 'Auth_Encoded',
            'Decision_Speed', 'Community_Encoded', 'Loyalty_Score', 'Total_Categories'
        ]
        
        regression_features = st.multiselect(
            "Select Features for Regression",
            regression_features_available,
            default=['Income_Encoded', 'Purchase_Encoded', 'Loyalty_Score', 'Auth_Encoded'],
            help="Choose features that influence willingness to pay"
        )
        
        if len(regression_features) < 2:
            st.warning("⚠️ Please select at least 2 features.")
            st.stop()
        
        # Prepare data
        X_reg = df_processed[regression_features].fillna(0)
        y_reg = df_processed[target_col].fillna(0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", int(len(X_reg) * (1 - test_size_reg)))
        with col2:
            st.metric("Test Samples", int(len(X_reg) * test_size_reg))
        with col3:
            st.metric("Target Mean", f"₹{y_reg.mean():,.0f}")
        
        # Model Selection
        st.subheader("🤖 Model Selection")
        
        model_options_reg = st.multiselect(
            "Select Regression Models to Train",
            [
                'Linear Regression',
                'Ridge Regression',
                'Lasso Regression',
                'ElasticNet',
                'Decision Tree',
                'Random Forest',
                'Gradient Boosting'
            ],
            default=['Linear Regression', 'Random Forest', 'Gradient Boosting']
        )
        
        if len(model_options_reg) == 0:
            st.warning("⚠️ Please select at least one model.")
            st.stop()
        
        # Train Models
        if st.button("🚀 Train Regression Models", type="primary", use_container_width=True):
            
            with st.spinner("🔄 Training regression models... Please wait..."):
                
                try:
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_reg, y_reg,
                        test_size=test_size_reg,
                        random_state=random_state_reg
                    )
                    
                    # Scale features
                    scaler_reg = StandardScaler()
                    X_train_scaled = scaler_reg.fit_transform(X_train)
                    X_test_scaled = scaler_reg.transform(X_test)
                    
                    # Initialize models
                    models_reg = {}
                    if 'Linear Regression' in model_options_reg:
                        models_reg['Linear Regression'] = LinearRegression()
                    if 'Ridge Regression' in model_options_reg:
                        models_reg['Ridge Regression'] = Ridge(alpha=1.0)
                    if 'Lasso Regression' in model_options_reg:
                        models_reg['Lasso Regression'] = Lasso(alpha=1.0)
                    if 'ElasticNet' in model_options_reg:
                        models_reg['ElasticNet'] = ElasticNet(alpha=1.0, l1_ratio=0.5)
                    if 'Decision Tree' in model_options_reg:
                        models_reg['Decision Tree'] = DecisionTreeRegressor(max_depth=10, random_state=random_state_reg)
                    if 'Random Forest' in model_options_reg:
                        models_reg['Random Forest'] = RandomForestRegressor(n_estimators=100, random_state=random_state_reg)
                    if 'Gradient Boosting' in model_options_reg:
                        models_reg['Gradient Boosting'] = GradientBoostingRegressor(n_estimators=100, random_state=random_state_reg)
                    
                    results_reg = {}
                    progress_bar = st.progress(0)
                    
                    for idx, (name, model) in enumerate(models_reg.items()):
                        
                        # Train
                        if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet']:
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                            
                            # CV
                            cv_scores = cross_val_score(
                                model, X_train_scaled, y_train,
                                cv=cv_folds_reg, scoring='neg_mean_squared_error'
                            )
                        else:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            
                            # CV
                            cv_scores = cross_val_score(
                                model, X_train, y_train,
                                cv=cv_folds_reg, scoring='neg_mean_squared_error'
                            )
                        
                        # Metrics
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        cv_rmse = np.sqrt(-cv_scores.mean())
                        
                        results_reg[name] = {
                            'RMSE': rmse,
                            'MAE': mae,
                            'R2': r2,
                            'CV_RMSE': cv_rmse,
                            'Predictions': y_pred,
                            'Model': model
                        }
                        
                        progress_bar.progress((idx + 1) / len(models_reg))
                    
                    progress_bar.empty()
                    
                    st.session_state.regression_results = {
                        'results': results_reg,
                        'X_test': X_test,
                        'y_test': y_test,
                        'X_train': X_train,
                        'y_train': y_train,
                        'scaler': scaler_reg,
                        'features': regression_features,
                        'category': category_name,
                        'X_all': X_reg,
                        'y_all': y_reg
                    }
                    
                    st.success("✅ All regression models trained successfully!")
                    
                    # Display Results
                    st.markdown("---")
                    st.subheader("📊 Model Performance Comparison")
                    
                    # Results Table
                    results_df_reg = pd.DataFrame({
                        'Model': list(results_reg.keys()),
                        'RMSE': [f"₹{results_reg[m]['RMSE']:,.0f}" for m in results_reg],
                        'MAE': [f"₹{results_reg[m]['MAE']:,.0f}" for m in results_reg],
                        'R² Score': [f"{results_reg[m]['R2']:.3f}" for m in results_reg],
                        'CV RMSE': [f"₹{results_reg[m]['CV_RMSE']:,.0f}" for m in results_reg]
                    })
                    
                    st.dataframe(results_df_reg, use_container_width=True, height=300)
                    
                    # Best Model
                    best_model_reg = max(results_reg.keys(), key=lambda k: results_reg[k]['R2'])
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🏆 Best Model", best_model_reg)
                    with col2:
                        st.metric("R² Score", f"{results_reg[best_model_reg]['R2']:.3f}")
                    with col3:
                        st.metric("RMSE", f"₹{results_reg[best_model_reg]['RMSE']:,.0f}")
                    with col4:
                        st.metric("MAE", f"₹{results_reg[best_model_reg]['MAE']:,.0f}")
                    
                    # Visualizations
                    st.markdown("---")
                    st.subheader("📈 Visualizations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Model Comparison
                        fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
                        
                        r2_values = [results_reg[m]['R2'] for m in results_reg]
                        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(r2_values)))
                        
                        ax_comp.barh(list(results_reg.keys()), r2_values, color=colors, edgecolor='black', linewidth=1.5)
                        ax_comp.set_xlabel('R² Score', fontsize=12, fontweight='bold')
                        ax_comp.set_title(f'{category_name}: R² Score Comparison', fontsize=14, fontweight='bold', pad=20)
                        ax_comp.set_xlim(0, 1)
                        ax_comp.grid(axis='x', alpha=0.3, linestyle='--')
                        
                        for i, (name, val) in enumerate(zip(results_reg.keys(), r2_values)):
                            ax_comp.text(val + 0.02, i, f'{val:.3f}', va='center', fontweight='bold')
                        
                        st.pyplot(fig_comp)
                        st.markdown(download_figure(fig_comp, "r2_comparison.png"), unsafe_allow_html=True)
                    
                    with col2:
                        # Actual vs Predicted
                        fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
                        
                        ax_pred.scatter(
                            y_test, results_reg[best_model_reg]['Predictions'],
                            alpha=0.6, s=50, edgecolors='black', linewidth=0.5
                        )
                        
                        min_val = min(y_test.min(), results_reg[best_model_reg]['Predictions'].min())
                        max_val = max(y_test.max(), results_reg[best_model_reg]['Predictions'].max())
                        
                        ax_pred.plot([min_val, max_val], [min_val, max_val],
                                    'r--', linewidth=2, label='Perfect Prediction')
                        
                        ax_pred.set_xlabel('Actual WTP (₹)', fontsize=12, fontweight='bold')
                        ax_pred.set_ylabel('Predicted WTP (₹)', fontsize=12, fontweight='bold')
                        ax_pred.set_title(f'{category_name}: Actual vs Predicted\n({best_model_reg})',
                                        fontsize=14, fontweight='bold', pad=20)
                        ax_pred.legend(fontsize=10)
                        ax_pred.grid(True, alpha=0.3, linestyle='--')
                        ax_pred.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/1000:.0f}K'))
                        ax_pred.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/1000:.0f}K'))
                        
                        st.pyplot(fig_pred)
                        st.markdown(download_figure(fig_pred, "actual_vs_predicted.png"), unsafe_allow_html=True)
                    
                    # Residual Plot
                    st.subheader("📉 Residual Analysis")
                    
                    residuals = y_test.values - results_reg[best_model_reg]['Predictions']
                    
                    fig_res, (ax_res1, ax_res2) = plt.subplots(1, 2, figsize=(16, 6))
                    
                    # Residual plot
                    ax_res1.scatter(results_reg[best_model_reg]['Predictions'], residuals,
                                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
                    ax_res1.axhline(y=0, color='r', linestyle='--', linewidth=2)
                    ax_res1.set_xlabel('Predicted WTP (₹)', fontsize=12, fontweight='bold')
                    ax_res1.set_ylabel('Residuals (₹)', fontsize=12, fontweight='bold')
                    ax_res1.set_title('Residual Plot', fontsize=14, fontweight='bold', pad=20)
                    ax_res1.grid(True, alpha=0.3, linestyle='--')
                    ax_res1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/1000:.0f}K'))
                    
                    # Residual histogram
                    ax_res2.hist(residuals, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
                    ax_res2.axvline(x=0, color='r', linestyle='--', linewidth=2)
                    ax_res2.set_xlabel('Residuals (₹)', fontsize=12, fontweight='bold')
                    ax_res2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                    ax_res2.set_title('Residual Distribution', fontsize=14, fontweight='bold', pad=20)
                    ax_res2.grid(axis='y', alpha=0.3, linestyle='--')
                    
                    plt.tight_layout()
                    st.pyplot(fig_res)
                    st.markdown(download_figure(fig_res, "residual_analysis.png"), unsafe_allow_html=True)
                    
                    # Feature Importance
                    if hasattr(results_reg[best_model_reg]['Model'], 'feature_importances_'):
                        st.markdown("---")
                        st.subheader("🔍 Feature Importance")
                        
                        importance_df = pd.DataFrame({
                            'Feature': regression_features,
                            'Importance': results_reg[best_model_reg]['Model'].feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                        colors_imp = plt.cm.plasma(np.linspace(0.3, 0.9, len(importance_df)))
                        ax_imp.barh(importance_df['Feature'], importance_df['Importance'],
                                   color=colors_imp, edgecolor='black', linewidth=1.5)
                        ax_imp.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
                        ax_imp.set_title(f'Feature Importance - {best_model_reg}',
                                        fontsize=14, fontweight='bold', pad=20)
                        ax_imp.grid(axis='x', alpha=0.3, linestyle='--')
                        
                        st.pyplot(fig_imp)
                        st.markdown(download_figure(fig_imp, "feature_importance_reg.png"), unsafe_allow_html=True)
                    
                    # Dynamic Pricing Engine
                    st.markdown("---")
                    st.subheader("💰 Dynamic Pricing Engine")
                    
                    st.markdown("""
                    <div class="warning-box">
                    <strong>Adjust pricing factors to generate personalized recommendations:</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        loyalty_discount = st.slider(
                            "Loyalty Discount (%)",
                            min_value=0,
                            max_value=20,
                            value=5,
                            step=1,
                            help="Discount for loyal customers"
                        )
                    
                    with col2:
                        premium_markup = st.slider(
                            "Premium Markup (%)",
                            min_value=0,
                            max_value=20,
                            value=5,
                            step=1,
                            help="Markup for high-income segments"
                        )
                    
                    with col3:
                        risk_discount = st.slider(
                            "Risk Discount (%)",
                            min_value=0,
                            max_value=20,
                            value=10,
                            step=1,
                            help="Discount for price-sensitive customers"
                        )
                    
                    with col4:
                        category_premium = st.slider(
                            "Category Interest Premium (%)",
                            min_value=0,
                            max_value=15,
                            value=5,
                            step=1,
                            help="Premium for high category interest"
                        )
                    
                    # Calculate dynamic prices
                    if best_model_reg in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet']:
                        X_all_scaled = scaler_reg.transform(X_reg)
                        base_predictions = results_reg[best_model_reg]['Model'].predict(X_all_scaled)
                    else:
                        base_predictions = results_reg[best_model_reg]['Model'].predict(X_reg)
                    
                    dynamic_prices = base_predictions.copy()
                    
                    # Apply adjustments
                    # High loyalty -> discount
                    high_loyalty = df_processed['Loyalty_Score'] >= 70
                    dynamic_prices[high_loyalty] *= (1 - loyalty_discount/100)
                    
                    # High income -> premium
                    high_income = df_processed['Income_Encoded'] >= 5
                    dynamic_prices[high_income] *= (1 + premium_markup/100)
                    
                    # Price sensitive -> discount
                    price_sensitive = df_processed['Q12_BiggestConcern'] == 'High prices'
                    dynamic_prices[price_sensitive] *= (1 - risk_discount/100)
                    
                    # High category interest -> premium
                    high_interest = df_processed['Total_Categories'] >= 3
                    dynamic_prices[high_interest] *= (1 + category_premium/100)
                    
                    # Ensure minimum price
                    dynamic_prices = np.maximum(dynamic_prices, base_predictions * 0.7)
                    
                    df_processed['Base_Price'] = base_predictions
                    df_processed['Dynamic_Price'] = dynamic_prices
                    df_processed['Adjustment_%'] = ((dynamic_prices - base_predictions) / base_predictions * 100)
                    
                    # Pricing Summary
                    st.subheader("📊 Pricing Distribution")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Avg Base Price", f"₹{base_predictions.mean():,.0f}")
                        st.metric("Avg Dynamic Price", f"₹{dynamic_prices.mean():,.0f}")
                    
                    with col2:
                        discount_pct = (dynamic_prices < base_predictions).sum() / len(dynamic_prices) * 100
                        premium_pct = (dynamic_prices > base_predictions).sum() / len(dynamic_prices) * 100
                        st.metric("Customers w/ Discount", f"{discount_pct:.1f}%")
                        st.metric("Customers w/ Premium", f"{premium_pct:.1f}%")
                    
                    with col3:
                        avg_adjustment = ((dynamic_prices - base_predictions) / base_predictions * 100).mean()
                        max_discount = ((base_predictions - dynamic_prices) / base_predictions * 100).max()
                        st.metric("Avg Adjustment", f"{avg_adjustment:+.1f}%")
                        st.metric("Max Discount", f"{max_discount:.1f}%")
                    
                    # Pricing Distribution Chart
                    fig_pricing, ax_pricing = plt.subplots(figsize=(14, 6))
                    
                    ax_pricing.hist(base_predictions, bins=40, alpha=0.5, label='Base Price',
                                   color='blue', edgecolor='black')
                    ax_pricing.hist(dynamic_prices, bins=40, alpha=0.5, label='Dynamic Price',
                                   color='green', edgecolor='black')
                    
                    ax_pricing.axvline(base_predictions.mean(), color='blue', linestyle='--',
                                      linewidth=2, label=f'Avg Base: ₹{base_predictions.mean():,.0f}')
                    ax_pricing.axvline(dynamic_prices.mean(), color='green', linestyle='--',
                                      linewidth=2, label=f'Avg Dynamic: ₹{dynamic_prices.mean():,.0f}')
                    
                    ax_pricing.set_xlabel('Price (₹)', fontsize=12, fontweight='bold')
                    ax_pricing.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                    ax_pricing.set_title(f'{category_name}: Price Distribution (Base vs Dynamic)',
                                        fontsize=14, fontweight='bold', pad=20)
                    ax_pricing.legend(fontsize=11, loc='upper right')
                    ax_pricing.grid(axis='y', alpha=0.3, linestyle='--')
                    ax_pricing.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/1000:.0f}K'))
                    
                    st.pyplot(fig_pricing)
                    st.markdown(download_figure(fig_pricing, "pricing_distribution.png"), unsafe_allow_html=True)
                    
                    # Sample Recommendations
                    st.markdown("---")
                    st.subheader("📋 Sample Pricing Recommendations")
                    
                    pricing_sample = df_processed[[
                        'ResponseID', 'Q1_Age', 'Q5_Income', 'Loyalty_Score',
                        'Base_Price', 'Dynamic_Price', 'Adjustment_%'
                    ]].head(20).copy()
                    
                    pricing_sample.columns = [
                        'Customer ID', 'Age', 'Income', 'Loyalty',
                        'Base Price', 'Dynamic Price', 'Adjustment %'
                    ]
                    
                    # Format
                    pricing_sample_formatted = pricing_sample.copy()
                    pricing_sample_formatted['Loyalty'] = pricing_sample_formatted['Loyalty'].apply(lambda x: f"{x:.0f}")
                    pricing_sample_formatted['Base Price'] = pricing_sample_formatted['Base Price'].apply(lambda x: f"₹{x:,.0f}")
                    pricing_sample_formatted['Dynamic Price'] = pricing_sample_formatted['Dynamic Price'].apply(lambda x: f"₹{x:,.0f}")
                    pricing_sample_formatted['Adjustment %'] = pricing_sample_formatted['Adjustment %'].apply(lambda x: f"{x:+.1f}%")
                    
                    st.dataframe(pricing_sample_formatted, use_container_width=True, height=400)
                    
                    # Download Results
                    st.markdown("---")
                    st.subheader("💾 Download Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        pred_export = pd.DataFrame({
                            'Customer_ID': df_processed['ResponseID'],
                            'Actual_WTP': y_reg.values,
                            'Predicted_WTP': base_predictions,
                            'Dynamic_Price': dynamic_prices,
                            'Adjustment_%': ((dynamic_prices - base_predictions) / base_predictions * 100)
                        })
                        
                        st.markdown(
                            create_download_link(pred_export, f"pricing_recommendations_{category_name.lower()}.csv",
                                               "Download Pricing Recommendations"),
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        st.markdown(
                            create_download_link(results_df_reg, "model_performance_regression.csv",
                                               "Download Model Performance"),
                            unsafe_allow_html=True
                        )
                    
                    with col3:
                        if hasattr(results_reg[best_model_reg]['Model'], 'feature_importances_'):
                            st.markdown(
                                create_download_link(importance_df, "feature_importance.csv",
                                                   "Download Feature Importance"),
                                unsafe_allow_html=True
                            )
                
                except Exception as e:
                    st.error(f"❌ Error during regression training: {str(e)}")
                    st.code(traceback.format_exc())
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h3>💎 Collectibles Platform Analytics Dashboard</h3>
        <p><strong>Built with ❤️ using Streamlit</strong></p>
        <p>Data-Driven Decision Making for Marketplace Success</p>
        <p style="margin-top: 10px; color: #999;">
            <small>
            Dashboard Features: Classification • Clustering • Association Rules • Regression • Dynamic Pricing
            </small>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
