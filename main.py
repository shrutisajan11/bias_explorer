import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Import the bias mitigation module
from bias_mitigation import (
    BiasMitigator, FairClassifier, PostProcessingMitigator,
    evaluate_mitigation_effectiveness, get_available_techniques,
    apply_mitigation_technique
)

st.set_page_config(page_title="Advanced Bias Explorer", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
.main-header {
    font-size: 3rem; font-weight: bold; text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem; border-radius: 10px; color: white; margin: 0.5rem 0;
}
.bias-alert {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0;
}
.fair-alert {
    background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
    padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0;
}
.sensitive-info {
    background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
    padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0;
}
.mitigation-info {
    background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
    padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0;
}
.improvement-alert {
    background: linear-gradient(135deg, #55a3ff 0%, #003d82 100%);
    padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

class EnhancedBiasExplorer:
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Industry-standard sensitive attributes mapping
        self.sensitive_attributes = {
            # Gender/Sex attributes
            'gender': ['gender', 'sex', 'male', 'female', 'gender_male', 'gender_female', 'is_male', 'is_female'],
            'race': ['race', 'ethnicity', 'ethnic', 'racial', 'white', 'black', 'asian', 'hispanic', 'latino', 
                    'african_american', 'race_white', 'race_black', 'race_asian', 'race_hispanic'],
            'age': ['age', 'age_group', 'age_category', 'senior', 'elderly', 'young', 'adult'],
            'religion': ['religion', 'religious', 'faith', 'christian', 'muslim', 'jewish', 'hindu', 'buddhist'],
            'disability': ['disability', 'disabled', 'handicap', 'impairment', 'medical_condition'],
            'sexual_orientation': ['sexual_orientation', 'lgbt', 'gay', 'lesbian', 'bisexual', 'heterosexual'],
            'marital_status': ['marital_status', 'married', 'single', 'divorced', 'widowed', 'spouse'],
            'nationality': ['nationality', 'country', 'citizen', 'immigrant', 'foreign', 'native'],
            'socioeconomic': ['income', 'salary', 'wealth', 'education', 'education_level', 'degree', 'college'],
            'geography': ['state', 'region', 'urban', 'rural', 'city', 'zip', 'postal', 'location']
        }
        
        # Initialize bias mitigator
        self.bias_mitigator = BiasMitigator()
        self.post_processor = PostProcessingMitigator()
        
    def identify_sensitive_attributes(self, df):
        """Identify potential sensitive attributes in the dataset"""
        sensitive_cols = {}
        df_cols_lower = [col.lower() for col in df.columns]
        
        for category, keywords in self.sensitive_attributes.items():
            found_cols = []
            for col in df.columns:
                col_lower = col.lower()
                # Check exact matches and partial matches
                if any(keyword in col_lower for keyword in keywords):
                    found_cols.append(col)
            
            if found_cols:
                sensitive_cols[category] = found_cols
        
        return sensitive_cols
    
    def load_sample_data(self):
        np.random.seed(42)
        n = 1000
        age = np.clip(np.random.normal(40, 12, n).astype(int), 18, 80)
        education = np.random.randint(1, 17, n)
        hours = np.clip(np.random.normal(40, 10, n), 1, 99)
        gender = np.random.choice(['Male', 'Female'], n, p=[0.7, 0.3])
        race = np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], n, p=[0.7, 0.15, 0.1, 0.05])
        
        income_prob = 0.3 + 0.01*age + 0.02*education + 0.005*hours
        income_prob = np.where(gender == 'Male', income_prob * 1.3, income_prob)
        income_prob = np.where(race == 'White', income_prob * 1.2, income_prob)
        income_prob = np.clip(income_prob, 0, 1)
        
        income = np.where(np.random.binomial(1, income_prob, n) == 1, '>50K', '<=50K')
        
        return pd.DataFrame({
            'age': age, 'education_num': education, 'hours_per_week': hours,
            'gender': gender, 'race': race, 'income': income
        })
    
    def preprocess_data(self, df, target_col):
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()
        
        le_dict = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            le_dict[col] = le
        
        if y.dtype == 'object':
            le_y = LabelEncoder()
            y = le_y.fit_transform(y)
            le_dict[target_col] = le_y
        
        return X, y, le_dict
    
    def calculate_fairness_metrics(self, y_true, y_pred, y_prob, sensitive_attr):
        if isinstance(sensitive_attr, np.ndarray):
            sensitive_attr = pd.Series(sensitive_attr)
        
        metrics = {}
        groups = sensitive_attr.unique()
        
        for group in groups:
            mask = sensitive_attr == group
            if np.sum(mask) == 0:
                continue
                
            yt, yp = y_true[mask], y_pred[mask]
            if len(yt) == 0:
                continue
            
            metrics[f'{group}_accuracy'] = accuracy_score(yt, yp)
            metrics[f'{group}_precision'] = precision_score(yt, yp, average='weighted', zero_division=0)
            metrics[f'{group}_recall'] = recall_score(yt, yp, average='weighted', zero_division=0)
            metrics[f'{group}_f1'] = f1_score(yt, yp, average='weighted', zero_division=0)
        
        # Demographic Parity
        pos_rates = {}
        for group in groups:
            mask = sensitive_attr == group
            if np.sum(mask) > 0:
                pos_rates[group] = np.mean(y_pred[mask])
        
        if len(pos_rates) > 1:
            metrics['demographic_parity_diff'] = max(pos_rates.values()) - min(pos_rates.values())
        
        # Equal Opportunity
        if len(np.unique(y_true)) == 2:
            tpr_by_group = {}
            for group in groups:
                mask = sensitive_attr == group
                if np.sum(mask) == 0:
                    continue
                yt, yp = y_true[mask], y_pred[mask]
                tp = np.sum((yt == 1) & (yp == 1))
                fn = np.sum((yt == 1) & (yp == 0))
                tpr_by_group[group] = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            if len(tpr_by_group) > 1:
                metrics['equal_opportunity_diff'] = max(tpr_by_group.values()) - min(tpr_by_group.values())
        
        return metrics, pos_rates
    
    def create_visualizations(self, metrics, pos_rates, sensitive_attr_name):
        groups = list(pos_rates.keys())
        
        # Performance comparison
        perf_data = []
        for group in groups:
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                if f'{group}_{metric}' in metrics:
                    perf_data.append({'Group': str(group), 'Metric': metric.capitalize(), 'Value': metrics[f'{group}_{metric}']})
        
        if perf_data:
            fig = px.bar(pd.DataFrame(perf_data), x='Metric', y='Value', color='Group',
                        title=f'Performance by {sensitive_attr_name}', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        # Positive rates
        pos_data = pd.DataFrame(list(pos_rates.items()), columns=[sensitive_attr_name, 'Rate'])
        fig2 = px.bar(pos_data, x=sensitive_attr_name, y='Rate', title=f'Positive Rates by {sensitive_attr_name}')
        st.plotly_chart(fig2, use_container_width=True)
        
        # Fairness metrics
        fairness_data = []
        for metric_name, key in [('Demographic Parity', 'demographic_parity_diff'), ('Equal Opportunity', 'equal_opportunity_diff')]:
            if key in metrics:
                fairness_data.append({
                    'Metric': metric_name, 'Value': metrics[key], 'Status': 'Fair' if metrics[key] < 0.1 else 'Biased'
                })
        
        if fairness_data:
            fig3 = go.Figure()
            for item in fairness_data:
                color = 'green' if item['Status'] == 'Fair' else 'red'
                fig3.add_trace(go.Bar(x=[item['Metric']], y=[item['Value']], marker_color=color, name=item['Status']))
            fig3.add_hline(y=0.1, line_dash="dash", line_color="gray")
            fig3.update_layout(title='Fairness Assessment', yaxis_title='Difference')
            st.plotly_chart(fig3, use_container_width=True)
        
        return fairness_data
    
    def create_comparison_visualization(self, original_metrics, mitigated_metrics, technique_name):
        """Create visualization comparing original vs mitigated results"""
        comparison_data = []
        
        for metric in ['demographic_parity_diff', 'equal_opportunity_diff']:
            if metric in original_metrics and metric in mitigated_metrics:
                comparison_data.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'Original': original_metrics[metric],
                    'Mitigated': mitigated_metrics[metric],
                    'Technique': technique_name
                })
        
        if comparison_data:
            df_comp = pd.DataFrame(comparison_data)
            df_melted = pd.melt(df_comp, id_vars=['Metric', 'Technique'], 
                              value_vars=['Original', 'Mitigated'], 
                              var_name='Type', value_name='Bias_Score')
            
            fig = px.bar(df_melted, x='Metric', y='Bias_Score', color='Type',
                        title=f'Bias Reduction: {technique_name}', barmode='group')
            fig.add_hline(y=0.1, line_dash="dash", line_color="gray", 
                         annotation_text="Fairness Threshold")
            st.plotly_chart(fig, use_container_width=True)
        
        return comparison_data

def main():
    st.markdown('<h1 class="main-header">⚖️ Advanced Bias Explorer</h1>', unsafe_allow_html=True)
    st.markdown("**Detect, Visualize, and Mitigate Bias in Machine Learning Models**")
    
    explorer = EnhancedBiasExplorer()
    
    # Sidebar configuration
    st.sidebar.title("🔧 Configuration")
    
    # Mode selection
    mode = st.sidebar.radio("Select Mode:", ["Bias Detection", "Bias Mitigation", "Compare Techniques"])
    
    data_option = st.sidebar.radio("Data source:", ["Use Sample Data", "Upload CSV"])
    
    df = None
    if data_option == "Use Sample Data":
        if st.sidebar.button("Load Sample Data"):
            df = explorer.load_sample_data()
            st.session_state.df = df
            st.sidebar.success("Data loaded!")
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV", type=['csv'])
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.session_state.df = df
                st.sidebar.success("Data uploaded!")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
    
    if 'df' in st.session_state:
        df = st.session_state.df
    
    if df is not None:
        # Identify sensitive attributes
        sensitive_cols = explorer.identify_sensitive_attributes(df)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
        with col2:
            st.subheader("Dataset Info")
            st.write(f"Shape: {df.shape[0]} × {df.shape[1]}")
        
        # Show identified sensitive attributes
        if sensitive_cols:
            st.markdown('<div class="sensitive-info"><h3>🎯 Detected Sensitive Attributes</h3>', unsafe_allow_html=True)
            for category, cols in sensitive_cols.items():
                st.markdown(f"**{category.replace('_', ' ').title()}:** {', '.join(cols)}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Flatten all sensitive columns for selection
            all_sensitive_cols = [col for cols in sensitive_cols.values() for col in cols]
        else:
            st.warning("⚠️ No industry-standard sensitive attributes detected in dataset")
            all_sensitive_cols = []
        
        st.sidebar.subheader("Model Configuration")
        target_col = st.sidebar.selectbox("Target column:", df.columns.tolist(), index=len(df.columns)-1)
        
        # Only show sensitive attributes for selection
        if all_sensitive_cols:
            available_sensitive = [c for c in all_sensitive_cols if c != target_col]
            if available_sensitive:
                sensitive_attr = st.sidebar.selectbox("Sensitive attribute:", available_sensitive)
            else:
                st.sidebar.error("No valid sensitive attributes available")
                return
        else:
            st.sidebar.error("No sensitive attributes detected. Upload dataset with standard attributes like gender, race, age, etc.")
            return
        
        model_name = st.sidebar.selectbox("Model:", list(explorer.models.keys()))
        features = st.sidebar.multiselect("Features:", [c for c in df.columns if c != target_col], 
                                        default=[c for c in df.columns if c != target_col])
        
        if mode == "Bias Detection":
            if st.sidebar.button("🚀 Train & Analyze"):
                if not features:
                    st.error("Select at least one feature!")
                    return
                
                try:
                    model_df = df[features + [target_col]].copy()
                    X, y, le_dict = explorer.preprocess_data(model_df, target_col)
                    sensitive_data = df[sensitive_attr].copy()
                    
                    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
                        X, y, sensitive_data, test_size=0.3, random_state=42, stratify=y
                    )
                    
                    if model_name == 'SVM':
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                    
                    model = explorer.models[model_name]
                    with st.spinner(f"Training {model_name}..."):
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        try:
                            y_prob = model.predict_proba(X_test)[:, 1]
                        except:
                            y_prob = None
                    
                    metrics, pos_rates = explorer.calculate_fairness_metrics(y_test, y_pred, y_prob, s_test)
                    
                    # Store results in session state
                    st.session_state.analysis_results = {
                        'model': model,
                        'X_train': X_train, 'X_test': X_test,
                        'y_train': y_train, 'y_test': y_test,
                        's_train': s_train, 's_test': s_test,
                        'y_pred': y_pred, 'y_prob': y_prob,
                        'metrics': metrics, 'pos_rates': pos_rates,
                        'model_name': model_name, 'sensitive_attr': sensitive_attr,
                        'scaler': scaler if model_name == 'SVM' else None
                    }
                    
                    st.subheader("📈 Results")
                    
                    # Overall metrics
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    for col, (name, val) in zip([col1, col2, col3, col4], 
                                              [('Accuracy', acc), ('Precision', prec), ('Recall', rec), ('F1', f1)]):
                        col.markdown(f'<div class="metric-container"><h3>{name}</h3><h2>{val:.3f}</h2></div>', 
                                   unsafe_allow_html=True)
                    
                    st.subheader("⚖️ Bias Analysis")
                    fairness_data = explorer.create_visualizations(metrics, pos_rates, sensitive_attr)
                    
                    if fairness_data:
                        bias_detected = any(item['Status'] == 'Biased' for item in fairness_data)
                        if bias_detected:
                            st.markdown('<div class="bias-alert"><h3>⚠️ Bias Detected!</h3><p>Model shows bias on industry-standard sensitive attribute. Consider using bias mitigation techniques.</p></div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="fair-alert"><h3>✅ Model Appears Fair</h3><p>Low bias detected across protected groups.</p></div>', unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        elif mode == "Bias Mitigation":
            st.sidebar.subheader("Mitigation Settings")
            
            # Get available techniques
            techniques = get_available_techniques()
            technique = st.sidebar.selectbox("Mitigation Technique:", list(techniques.keys()))
            
            # Technique-specific parameters
            if technique == 'resampling':
                strategy = st.sidebar.selectbox("Resampling Strategy:", ['undersample', 'oversample'])
            elif technique == 'fairness_constraints':
                lambda_fairness = st.sidebar.slider("Fairness Lambda:", 0.1, 2.0, 1.0, 0.1)
            elif technique == 'threshold_optimization':
                fairness_metric = st.sidebar.selectbox("Fairness Metric:", ['equal_opportunity', 'demographic_parity'])
            elif technique == 'calibration':
                calibration_method = st.sidebar.selectbox("Calibration Method:", ['platt', 'isotonic'])
            
            if st.sidebar.button("🔧 Apply Mitigation"):
                if not features:
                    st.error("Select at least one feature!")
                    return
                
                try:
                    model_df = df[features + [target_col]].copy()
                    X, y, le_dict = explorer.preprocess_data(model_df, target_col)
                    sensitive_data = df[sensitive_attr].copy()
                    
                    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
                        X, y, sensitive_data, test_size=0.3, random_state=42, stratify=y
                    )
                    
                    if model_name == 'SVM':
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                    
                    # Train original model for comparison
                    original_model = explorer.models[model_name]
                    original_model.fit(X_train, y_train)
                    y_pred_original = original_model.predict(X_test)
                    
                    # Apply mitigation technique
                    kwargs = {}
                    if technique == 'resampling':
                        kwargs['strategy'] = strategy
                    elif technique == 'fairness_constraints':
                        kwargs['lambda_fairness'] = lambda_fairness
                    elif technique == 'threshold_optimization':
                        kwargs['fairness_metric'] = fairness_metric
                    elif technique == 'calibration':
                        kwargs['calibration_method'] = calibration_method
                    
                    model_copy = explorer.models[model_name]
                    y_pred_mitigated, mitigation_info = apply_mitigation_technique(
                        technique, X_train, y_train, s_train, X_test, y_test, s_test, model_copy, **kwargs
                    )
                    
                    # Calculate metrics for both models
                    original_metrics, original_pos_rates = explorer.calculate_fairness_metrics(
                        y_test, y_pred_original, None, s_test
                    )
                    mitigated_metrics, mitigated_pos_rates = explorer.calculate_fairness_metrics(
                        y_test, y_pred_mitigated, None, s_test
                    )
                    
                    # Evaluate effectiveness
                    effectiveness = evaluate_mitigation_effectiveness(
                        y_test, y_pred_original, y_pred_mitigated, s_test, techniques[technique]
                    )
                    
                    st.subheader("🔧 Mitigation Results")
                    
                    # Show technique info
                    st.markdown(f'<div class="mitigation-info"><h3>Applied Technique: {techniques[technique]}</h3><p>Technique details: {mitigation_info}</p></div>', unsafe_allow_html=True)
                    
                    # Performance comparison
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original Model")
                        acc_orig = accuracy_score(y_test, y_pred_original)
                        st.metric("Accuracy", f"{acc_orig:.3f}")
                        if 'demographic_parity_diff' in original_metrics:
                            st.metric("Demographic Parity Diff", f"{original_metrics['demographic_parity_diff']:.3f}")
                        if 'equal_opportunity_diff' in original_metrics:
                            st.metric("Equal Opportunity Diff", f"{original_metrics['equal_opportunity_diff']:.3f}")
                    
                    with col2:
                        st.subheader("Mitigated Model")
                        acc_mit = accuracy_score(y_test, y_pred_mitigated)
                        st.metric("Accuracy", f"{acc_mit:.3f}", delta=f"{acc_mit - acc_orig:.3f}")
                        if 'demographic_parity_diff' in mitigated_metrics:
                            dp_improvement = original_metrics.get('demographic_parity_diff', 0) - mitigated_metrics['demographic_parity_diff']
                            st.metric("Demographic Parity Diff", f"{mitigated_metrics['demographic_parity_diff']:.3f}", 
                                     delta=f"{-dp_improvement:.3f}")
                        if 'equal_opportunity_diff' in mitigated_metrics:
                            eo_improvement = original_metrics.get('equal_opportunity_diff', 0) - mitigated_metrics['equal_opportunity_diff']
                            st.metric("Equal Opportunity Diff", f"{mitigated_metrics['equal_opportunity_diff']:.3f}", 
                                     delta=f"{-eo_improvement:.3f}")
                    
                    # Visualize comparison
                    explorer.create_comparison_visualization(original_metrics, mitigated_metrics, techniques[technique])
                    
                    # Show improvement summary
                    if effectiveness['improvement']:
                        improvements = []
                        for metric, improvement in effectiveness['improvement'].items():
                            if improvement > 0:
                                improvements.append(f"{metric.replace('_', ' ').title()}: {improvement:.1f}% improvement")
                        
                        if improvements:
                            improvement_text = "<br>".join(improvements)
                            accuracy_change = effectiveness['accuracy_tradeoff']
                            st.markdown(f'<div class="improvement-alert"><h3>✨ Improvement Summary</h3><p>{improvement_text}<br><br><strong>Accuracy Trade-off:</strong> {accuracy_change:+.3f}</p></div>', unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error during mitigation: {str(e)}")
        
        elif mode == "Compare Techniques":
            st.sidebar.subheader("Comparison Settings")
            techniques_to_compare = st.sidebar.multiselect(
                "Select Techniques to Compare:", 
                list(get_available_techniques().keys()),
                default=['resampling', 'threshold_optimization']
            )
            
            if st.sidebar.button("🔍 Compare Techniques"):
                if not features or not techniques_to_compare:
                    st.error("Select features and at least one technique!")
                    return
                
                try:
                    model_df = df[features + [target_col]].copy()
                    X, y, le_dict = explorer.preprocess_data(model_df, target_col)
                    sensitive_data = df[sensitive_attr].copy()
                    
                    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
                        X, y, sensitive_data, test_size=0.3, random_state=42, stratify=y
                    )
                    
                    if model_name == 'SVM':
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                    
                    # Train original model
                    original_model = explorer.models[model_name]
                    original_model.fit(X_train, y_train)
                    y_pred_original = original_model.predict(X_test)
                    
                    original_metrics, _ = explorer.calculate_fairness_metrics(y_test, y_pred_original, None, s_test)
                    
                    st.subheader("🏆 Technique Comparison")
                    
                    comparison_results = []
                    
                    # Test each technique
                    for tech in techniques_to_compare:
                        try:
                            model_copy = explorer.models[model_name]
                            y_pred_tech, _ = apply_mitigation_technique(
                                tech, X_train, y_train, s_train, X_test, y_test, s_test, model_copy
                            )
                            
                            tech_metrics, _ = explorer.calculate_fairness_metrics(y_test, y_pred_tech, None, s_test)
                            
                            # Calculate improvements
                            dp_improvement = 0
                            eo_improvement = 0
                            
                            if 'demographic_parity_diff' in original_metrics and 'demographic_parity_diff' in tech_metrics:
                                dp_improvement = ((original_metrics['demographic_parity_diff'] - tech_metrics['demographic_parity_diff']) / original_metrics['demographic_parity_diff']) * 100
                            
                            if 'equal_opportunity_diff' in original_metrics and 'equal_opportunity_diff' in tech_metrics:
                                eo_improvement = ((original_metrics['equal_opportunity_diff'] - tech_metrics['equal_opportunity_diff']) / original_metrics['equal_opportunity_diff']) * 100
                            
                            accuracy_change = accuracy_score(y_test, y_pred_tech) - accuracy_score(y_test, y_pred_original)
                            
                            comparison_results.append({
                                'Technique': get_available_techniques()[tech],
                                'DP_Improvement': dp_improvement,
                                'EO_Improvement': eo_improvement,
                                'Accuracy_Change': accuracy_change,
                                'Overall_Score': (dp_improvement + eo_improvement) / 2 - abs(accuracy_change) * 100
                            })
                        
                        except Exception as e:
                            st.warning(f"Could not evaluate {tech}: {str(e)}")
                    
                    if comparison_results:
                        # Create comparison table
                        df_comparison = pd.DataFrame(comparison_results)
                        df_comparison = df_comparison.sort_values('Overall_Score', ascending=False)
                        
                        st.dataframe(df_comparison.round(3), use_container_width=True)
                        
                        # Visualize comparison
                        fig_comp = px.bar(df_comparison, x='Technique', y=['DP_Improvement', 'EO_Improvement'],
                                         title='Bias Reduction Comparison (%)', barmode='group')
                        st.plotly_chart(fig_comp, use_container_width=True)
                        
                        # Show best technique
                        best_technique = df_comparison.iloc[0]['Technique']
                        st.markdown(f'<div class="improvement-alert"><h3>🏆 Best Technique: {best_technique}</h3><p>This technique showed the best overall performance considering both bias reduction and accuracy preservation.</p></div>', unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error during comparison: {str(e)}")
    
    else:
        st.markdown("""
        ## Welcome to Advanced Bias Explorer! 👋
        
        **Enhanced Features:**
        - **Bias Detection**: Automatically detect sensitive attributes and measure bias
        - **Bias Mitigation**: Apply industry-standard mitigation techniques
        - **Technique Comparison**: Compare multiple mitigation approaches
        
        **Available Mitigation Techniques:**
        - **Pre-processing**: Data resampling, instance reweighting
        - **In-processing**: Fairness constraints, adversarial debiasing
        - **Post-processing**: Threshold optimization, prediction calibration
        
        **Get Started:**
        1. Load sample data or upload your CSV
        2. Choose your analysis mode
        3. Configure model settings
        4. Run analysis or apply mitigation
        
        **You'll Get:**
        - Comprehensive bias assessment
        - Multiple mitigation options
        - Performance vs fairness trade-offs
        - Actionable recommendations
        """)

if __name__ == "__main__":
    main()