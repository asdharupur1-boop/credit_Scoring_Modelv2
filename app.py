# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys
import base64
from io import BytesIO

class AICreditScoring2:
    def __init__(self):
        self.bureau_models = {}
        self.ai_scoring_models = {}
        self.bureau_artifacts = {}
        self.ai_scoring_artifacts = {}
        self.loaded = False
        self.current_prediction = None
        
    def load_all_models(self):
        """Load all models from both systems"""
        try:
            # Initialize demo models since we don't have actual model files
            self._initialize_demo_models()
            self.loaded = True
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            return False

    def _initialize_demo_models(self):
        """Initialize demo models for testing"""
        # Demo bureau models
        self.bureau_models = {
            'xgboost': 'demo_model',
            'lightgbm': 'demo_model', 
            'random_forest': 'demo_model',
            'logistic': 'demo_model'
        }
        
        # Demo AI scoring models
        self.ai_scoring_models = {
            'xgboost': 'demo_model',
            'lightgbm': 'demo_model',
            'scorecard': 'demo_model',
            'scorecard_system': 'demo_model'
        }
        
        print("✅ Demo Models Initialized")

    def setup_application(self):
        """Setup the enterprise application"""
        st.set_page_config(
            page_title="AI Credit Scoring 2.0 | Enterprise Platform",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # Custom CSS for enterprise look
        st.markdown("""
        <style>
            .main-header {
                font-size: 2.5rem;
                background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
                margin-bottom: 1rem;
                font-weight: 800;
            }
            .enterprise-card {
                background: white;
                padding: 1rem;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                border-left: 4px solid #4ECDC4;
                margin: 0.5rem 0;
                word-wrap: break-word;
                overflow: hidden;
            }
            .model-badge {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 0.4rem 0.8rem;
                border-radius: 16px;
                font-size: 0.75rem;
                font-weight: 600;
                margin: 0.1rem;
                display: inline-block;
            }
            .impact-metric {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                padding: 1rem;
                border-radius: 12px;
                text-align: center;
                height: 120px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                word-wrap: break-word;
            }
            .risk-high {
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
                color: white;
                padding: 1rem;
                border-radius: 12px;
                text-align: center;
            }
            .risk-medium {
                background: linear-gradient(135deg, #ffd93d 0%, #ffcd3c 100%);
                color: black;
                padding: 1rem;
                border-radius: 12px;
                text-align: center;
            }
            .risk-low {
                background: linear-gradient(135deg, #4ECDC4 0%, #44a08d 100%);
                color: white;
                padding: 1rem;
                border-radius: 12px;
                text-align: center;
            }
            .stButton button {
                width: 100%;
                word-wrap: break-word;
                white-space: normal;
            }
            .section-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
            }
            .pdf-section {
                background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                padding: 2rem;
                border-radius: 12px;
                margin: 1rem 0;
            }
        </style>
        """, unsafe_allow_html=True)

    def render_enterprise_header(self):
        """Render enterprise header"""
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 class="main-header">🚀 AI Credit Scoring 2.0</h1>
            <p style="font-size: 1.1rem; color: #666; margin-bottom: 0.8rem;">
            Advanced AI-Powered Credit Risk Assessment Platform
            </p>
            <div style="display: flex; justify-content: center; gap: 0.5rem; flex-wrap: wrap;">
                <span class="model-badge">🤖 Multi-Model AI</span>
                <span class="model-badge">🎯 Real-time Scoring</span>
                <span class="model-badge">📊 Deep Analytics</span>
                <span class="model-badge">📄 PDF Reports</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def predict_ensemble(self, features, system='both'):
        """Make predictions using ensemble of models"""
        # Demo prediction logic
        credit_score = features[0]
        annual_income = features[1]
        dti_ratio = features[3]
        employment_length = features[2]
        derogatory_marks = features[6]
        
        # Enhanced scoring algorithm
        base_score = (
            credit_score * 0.5 + 
            (min(annual_income, 5000000) / 50000) * 0.15 +
            (employment_length * 10) * 0.1 +
            ((1 - dti_ratio) * 100) * 0.15 -
            (derogatory_marks * 20) * 0.1
        )
        
        # Normalize to 300-850 range
        final_score = max(300, min(850, base_score))
        
        # Calculate default probability (more realistic)
        default_prob = max(0.01, min(0.99, (850 - final_score) / 550 * 0.8 + np.random.normal(0, 0.05)))
        
        predictions = {
            'bureau': {
                'xgboost': default_prob,
                'lightgbm': default_prob * 0.95,
                'random_forest': default_prob * 1.05,
                'logistic': default_prob * 0.9,
                'ensemble': default_prob
            },
            'ai_scoring': {
                'xgboost': default_prob * 0.98,
                'lightgbm': default_prob * 0.96,
                'scorecard': default_prob * 1.02,
                'ensemble': default_prob
            },
            'final_score': final_score,
            'risk_level': self.get_risk_level(final_score),
            'features': features,
            'timestamp': datetime.now()
        }
        
        return predictions

    def get_risk_level(self, score):
        """Get risk level based on credit score"""
        if score >= 750:
            return "LOW RISK"
        elif score >= 650:
            return "MEDIUM RISK"
        else:
            return "HIGH RISK"

    def render_credit_scoring_form(self):
        """Render the main credit scoring form"""
        st.markdown('<div class="section-header"><h2>📋 Applicant Credit Information</h2></div>', unsafe_allow_html=True)
        
        with st.form("credit_application"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("👤 Personal & Financial Info")
                credit_score = st.slider("Credit Score", 300, 850, 720, help="FICO credit score range 300-850")
                annual_income = st.number_input("Annual Income (₹)", 100000, 5000000, 750000, 50000, help="Gross annual income in INR")
                employment_length = st.number_input("Employment Length (years)", 0, 40, 5, help="Total years in current employment")
                total_accounts = st.number_input("Total Credit Accounts", 1, 20, 8, help="Number of active credit accounts")
                
            with col2:
                st.subheader("📊 Credit Metrics")
                dti_ratio = st.slider("Debt-to-Income Ratio", 0.1, 0.8, 0.35, 0.01, help="Monthly debt payments / Monthly income")
                credit_utilization = st.slider("Credit Utilization Ratio", 0.0, 1.0, 0.3, 0.01, help="Total credit used / Total credit limit")
                derogatory_marks = st.number_input("Derogatory Marks", 0, 10, 0, help="Number of late payments, defaults, etc.")
                savings_balance = st.number_input("Savings Balance (₹)", 0, 2000000, 200000, 10000, help="Total savings and investments")
            
            # System selection
            st.subheader("🤖 AI System Selection")
            system_choice = st.radio(
                "Choose AI System for Analysis:",
                ["Both Systems (Recommended)", "Credit Bureau Only", "AI Scoring 2.0 Only"],
                horizontal=True
            )
            
            # Submit button
            submitted = st.form_submit_button("🚀 Analyze Credit Risk", use_container_width=True)
            
            if submitted:
                features = [credit_score, annual_income, employment_length, dti_ratio, 
                          credit_utilization, total_accounts, derogatory_marks, savings_balance]
                
                with st.spinner("🤖 AI Systems Analyzing Credit Risk..."):
                    system_map = {
                        "Both Systems (Recommended)": "both",
                        "Credit Bureau Only": "bureau", 
                        "AI Scoring 2.0 Only": "ai_scoring"
                    }
                    
                    self.current_prediction = self.predict_ensemble(features, system_map[system_choice])
                    st.success("✅ Credit Assessment Complete!")
                    return True
        return False

    def render_scoring_results(self):
        """Render comprehensive scoring results"""
        if not self.current_prediction:
            return
            
        st.markdown('<div class="section-header"><h2>🎯 Credit Assessment Results</h2></div>', unsafe_allow_html=True)
        
        # Overall Score Card
        final_score = self.current_prediction['final_score']
        risk_level = self.current_prediction['risk_level']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏦 Bureau Score", f"{300 + self.current_prediction['bureau']['ensemble'] * 550:.0f}")
        with col2:
            st.metric("🚀 AI 2.0 Score", f"{300 + self.current_prediction['ai_scoring']['ensemble'] * 550:.0f}")
        with col3:
            st.metric("📊 Final Credit Score", f"{final_score:.0f}")
        with col4:
            default_prob = self.current_prediction['bureau']['ensemble']
            st.metric("📉 Default Probability", f"{default_prob:.1%}")
        
        # Risk Assessment
        risk_class = "risk-low" if risk_level == "LOW RISK" else "risk-medium" if risk_level == "MEDIUM RISK" else "risk-high"
        st.markdown(f"""
        <div class="{risk_class}">
            <h3 style="margin: 0; text-align: center;">{risk_level}</h3>
            <p style="margin: 0; text-align: center; font-size: 14px;">Credit Risk Assessment</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendation
        st.subheader("💡 Credit Decision & Recommendations")
        
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            if risk_level == "LOW RISK":
                st.success("""
                **✅ APPROVAL RECOMMENDED**
                - **Credit Limit:** ₹8,00,000
                - **Interest Rate:** 10.5% p.a.
                - **Loan Term:** 36 months
                - **Processing:** Fast-track
                """)
            elif risk_level == "MEDIUM RISK":
                st.warning("""
                **⚠️ CONDITIONAL APPROVAL**
                - **Credit Limit:** ₹4,00,000  
                - **Interest Rate:** 14.5% p.a.
                - **Loan Term:** 24 months
                - **Requirements:** Additional collateral
                """)
            else:
                st.error("""
                **🔴 FURTHER REVIEW REQUIRED**
                - **Credit Limit:** ₹1,50,000
                - **Interest Rate:** 18.5% p.a.
                - **Loan Term:** 12 months
                - **Requirements:** Strong collateral + guarantor
                """)
        
        with rec_col2:
            st.info("""
            **🛡️ Risk Mitigation Plan**
            - Credit behavior monitoring (6 months)
            - Payment pattern tracking
            - Limit review after 12 months
            - Financial counseling sessions
            - Progressive limit increases
            """)

    def render_model_analytics(self):
        """Render detailed model analytics"""
        if not self.current_prediction:
            return
            
        st.markdown('<div class="section-header"><h2>📊 AI Model Analytics</h2></div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["🤖 Model Performance", "🎯 Feature Analysis", "📈 Comparison"])
        
        with tab1:
            # Model Performance Comparison
            st.subheader("Model Performance Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bureau Models Performance
                bureau_models = list(self.current_prediction['bureau'].keys())[:-1]  # Exclude ensemble
                bureau_scores = [300 + score * 550 for score in list(self.current_prediction['bureau'].values())[:-1]]
                
                fig1 = px.bar(
                    x=bureau_models, y=bureau_scores,
                    title="Credit Bureau Models - Score Output",
                    labels={'x': 'Models', 'y': 'Credit Score'},
                    color=bureau_scores,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # AI Scoring Models Performance
                ai_models = list(self.current_prediction['ai_scoring'].keys())[:-1]
                ai_scores = [300 + score * 550 for score in list(self.current_prediction['ai_scoring'].values())[:-1]]
                
                fig2 = px.bar(
                    x=ai_models, y=ai_scores,
                    title="AI Scoring 2.0 Models - Score Output",
                    labels={'x': 'Models', 'y': 'Credit Score'},
                    color=ai_scores,
                    color_continuous_scale='Plasma'
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            # Feature Importance Visualization
            st.subheader("Feature Importance Analysis")
            
            features = ['Credit Score', 'Annual Income', 'Employment', 'DTI Ratio', 
                       'Credit Utilization', 'Total Accounts', 'Derogatory Marks', 'Savings']
            importance = [0.32, 0.18, 0.12, 0.15, 0.08, 0.07, 0.05, 0.03]
            
            fig = px.bar(
                x=importance, y=features, orientation='h',
                title="Feature Importance in Credit Decision",
                labels={'x': 'Importance Weight', 'y': 'Features'},
                color=importance,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature correlation matrix
            st.subheader("Feature Correlation Heatmap")
            corr_matrix = np.array([
                [1.00, 0.45, 0.35, -0.25, -0.30, 0.20, -0.40, 0.30],
                [0.45, 1.00, 0.60, -0.15, -0.20, 0.25, -0.25, 0.55],
                [0.35, 0.60, 1.00, -0.10, -0.15, 0.15, -0.20, 0.40],
                [-0.25, -0.15, -0.10, 1.00, 0.45, -0.10, 0.35, -0.15],
                [-0.30, -0.20, -0.15, 0.45, 1.00, -0.15, 0.40, -0.10],
                [0.20, 0.25, 0.15, -0.10, -0.15, 1.00, -0.15, 0.20],
                [-0.40, -0.25, -0.20, 0.35, 0.40, -0.15, 1.00, -0.25],
                [0.30, 0.55, 0.40, -0.15, -0.10, 0.20, -0.25, 1.00]
            ])
            
            fig_corr = px.imshow(
                corr_matrix,
                x=features,
                y=features,
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu_r',
                aspect="auto"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with tab3:
            # Model comparison
            st.subheader("Model Score Distribution")
            
            all_models = []
            all_scores = []
            all_types = []
            
            # Bureau models
            for model, score in self.current_prediction['bureau'].items():
                if model != 'ensemble':
                    all_models.append(f"Bureau_{model}")
                    all_scores.append(300 + score * 550)
                    all_types.append("Bureau")
            
            # AI models
            for model, score in self.current_prediction['ai_scoring'].items():
                if model != 'ensemble':
                    all_models.append(f"AI_{model}")
                    all_scores.append(300 + score * 550)
                    all_types.append("AI 2.0")
            
            comparison_df = pd.DataFrame({
                'Model': all_models,
                'Score': all_scores,
                'Type': all_types
            })
            
            fig_compare = px.box(
                comparison_df, x='Type', y='Score',
                title="Model Score Distribution by System",
                color='Type',
                points="all"
            )
            st.plotly_chart(fig_compare, use_container_width=True)

    def render_portfolio_analytics(self):
        """Render portfolio-level analytics"""
        if not self.current_prediction:
            return
            
        st.markdown('<div class="section-header"><h2>📈 Portfolio Risk Analytics</h2></div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📊 Risk Distribution", "📈 Trends", "👥 Segmentation"])
        
        with tab1:
            # Risk Distribution
            col1, col2 = st.columns(2)
            
            with col1:
                risk_data = {
                    'Risk Level': ['Low Risk (750+)', 'Medium Risk (650-749)', 'High Risk (<650)'],
                    'Percentage': [65, 25, 10],
                    'Count': [32600, 12500, 5147],
                    'Default Rate': [0.8, 2.9, 8.7]
                }
                
                fig_pie = px.pie(
                    risk_data, values='Percentage', names='Risk Level',
                    title="Portfolio Risk Distribution",
                    color_discrete_sequence=['#4ECDC4', '#FFD93D', '#FF6B6B']
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Default rates by risk category
                fig_bar = px.bar(
                    risk_data, x='Risk Level', y='Default Rate',
                    title="Default Rates by Risk Category",
                    color='Risk Level',
                    color_discrete_sequence=['#4ECDC4', '#FFD93D', '#FF6B6B'],
                    text='Default Rate'
                )
                fig_bar.update_traces(texttemplate='%{text}%', textposition='outside')
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab2:
            # Performance trends
            st.subheader("Portfolio Performance Trends")
            
            trend_data = {
                'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug'],
                'Approval Rate': [78, 82, 85, 83, 86, 88, 87, 89],
                'Default Rate': [3.2, 2.8, 2.5, 2.3, 2.1, 1.9, 1.8, 1.7],
                'Avg Score': [715, 718, 722, 724, 726, 728, 729, 731]
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_trend1 = px.line(
                    trend_data, x='Month', y=['Approval Rate', 'Default Rate'],
                    title="Approval vs Default Rates",
                    labels={'value': 'Percentage', 'variable': 'Metric'}
                )
                st.plotly_chart(fig_trend1, use_container_width=True)
            
            with col2:
                fig_trend2 = px.line(
                    trend_data, x='Month', y='Avg Score',
                    title="Average Credit Score Trend",
                    labels={'y': 'Credit Score'}
                )
                st.plotly_chart(fig_trend2, use_container_width=True)
        
        with tab3:
            # Customer Segmentation
            st.subheader("Customer Segmentation Analysis")
            
            segment_data = {
                'Segment': ['Prime (750+)', 'Near Prime (650-749)', 'Subprime (300-649)'],
                'Customers': [25600, 18700, 5947],
                'Avg Income': ['₹12.5L', '₹8.2L', '₹5.1L'],
                'Avg Age': ['42 years', '38 years', '35 years'],
                'Default Rate': ['0.8%', '2.9%', '8.7%'],
                'Avg Utilization': ['28%', '45%', '68%']
            }
            
            st.dataframe(pd.DataFrame(segment_data), use_container_width=True)
            
            # Segmentation visualization
            fig_segment = px.sunburst(
                pd.DataFrame(segment_data),
                path=['Segment'],
                values='Customers',
                title="Customer Distribution by Segment",
                color='Customers',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_segment, use_container_width=True)

    def generate_html_report(self):
        """Generate HTML report as PDF alternative"""
        if not self.current_prediction:
            return None
            
        # Create comprehensive HTML report
        features = self.current_prediction['features']
        final_score = self.current_prediction['final_score']
        risk_level = self.current_prediction['risk_level']
        timestamp = self.current_prediction['timestamp']
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Credit Scoring 2.0 - Credit Assessment Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 20px; }}
                .section {{ margin: 30px 0; }}
                .metric {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; }}
                .risk-low {{ background: #d4edda; color: #155724; padding: 15px; border-radius: 8px; }}
                .risk-medium {{ background: #fff3cd; color: #856404; padding: 15px; border-radius: 8px; }}
                .risk-high {{ background: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 AI Credit Scoring 2.0</h1>
                <h2>Credit Assessment Report</h2>
                <p>Generated on: {timestamp.strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="section">
                <h3>📋 Applicant Information</h3>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Credit Score</td><td>{features[0]}</td></tr>
                    <tr><td>Annual Income</td><td>₹{features[1]:,}</td></tr>
                    <tr><td>Employment Length</td><td>{features[2]} years</td></tr>
                    <tr><td>DTI Ratio</td><td>{features[3]:.1%}</td></tr>
                    <tr><td>Credit Utilization</td><td>{features[4]:.1%}</td></tr>
                    <tr><td>Total Accounts</td><td>{features[5]}</td></tr>
                    <tr><td>Derogatory Marks</td><td>{features[6]}</td></tr>
                    <tr><td>Savings Balance</td><td>₹{features[7]:,}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h3>🎯 Assessment Results</h3>
                <div class="metric">
                    <h4>Final Credit Score: {final_score:.0f}</h4>
                    <div class="risk-{risk_level.lower().split()[0]}">
                        <strong>Risk Level: {risk_level}</strong>
                    </div>
                </div>
                
                <table>
                    <tr><th>Model System</th><th>Score</th><th>Default Probability</th></tr>
                    <tr><td>Credit Bureau Ensemble</td><td>{300 + self.current_prediction['bureau']['ensemble'] * 550:.0f}</td><td>{self.current_prediction['bureau']['ensemble']:.1%}</td></tr>
                    <tr><td>AI Scoring 2.0 Ensemble</td><td>{300 + self.current_prediction['ai_scoring']['ensemble'] * 550:.0f}</td><td>{self.current_prediction['ai_scoring']['ensemble']:.1%}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h3>💡 Recommendations</h3>
        """
        
        if risk_level == "LOW RISK":
            html_content += """
                <div class="risk-low">
                    <h4>✅ APPROVAL RECOMMENDED</h4>
                    <ul>
                        <li>Credit Limit: ₹8,00,000</li>
                        <li>Interest Rate: 10.5% p.a.</li>
                        <li>Loan Term: 36 months</li>
                        <li>Processing: Fast-track</li>
                    </ul>
                </div>
            """
        elif risk_level == "MEDIUM RISK":
            html_content += """
                <div class="risk-medium">
                    <h4>⚠️ CONDITIONAL APPROVAL</h4>
                    <ul>
                        <li>Credit Limit: ₹4,00,000</li>
                        <li>Interest Rate: 14.5% p.a.</li>
                        <li>Loan Term: 24 months</li>
                        <li>Requirements: Additional collateral</li>
                    </ul>
                </div>
            """
        else:
            html_content += """
                <div class="risk-high">
                    <h4>🔴 FURTHER REVIEW REQUIRED</h4>
                    <ul>
                        <li>Credit Limit: ₹1,50,000</li>
                        <li>Interest Rate: 18.5% p.a.</li>
                        <li>Loan Term: 12 months</li>
                        <li>Requirements: Strong collateral + guarantor</li>
                    </ul>
                </div>
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h3>📊 Model Details</h3>
                <table>
                    <tr><th>Model</th><th>Type</th><th>Score</th></tr>
        """
        
        # Add bureau models
        for model, score in self.current_prediction['bureau'].items():
            if model != 'ensemble':
                html_content += f"<tr><td>{model.upper()}</td><td>Bureau</td><td>{300 + score * 550:.0f}</td></tr>"
        
        # Add AI models
        for model, score in self.current_prediction['ai_scoring'].items():
            if model != 'ensemble':
                html_content += f"<tr><td>{model.upper()}</td><td>AI 2.0</td><td>{300 + score * 550:.0f}</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <p><em>Report generated by AI Credit Scoring 2.0 Enterprise Platform</em></p>
                <p><em>Confidential - For internal use only</em></p>
            </div>
        </body>
        </html>
        """
        
        return html_content

    def render_pdf_report_section(self):
        """Render PDF report download section using HTML alternative"""
        st.markdown('<div class="section-header"><h2>📄 Download Comprehensive Report</h2></div>', unsafe_allow_html=True)
        
        if self.current_prediction:
            # Generate HTML report
            html_content = self.generate_html_report()
            
            if html_content:
                # Create download button for HTML report
                st.markdown("""
                <div class="pdf-section">
                    <h3 style="text-align: center; color: #2c3e50;">📋 Comprehensive Credit Assessment Report</h3>
                    <p style="text-align: center;">Download a detailed report containing all assessment results, analytics, and recommendations.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Convert HTML to bytes for download
                html_bytes = html_content.encode('utf-8')
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        label="📥 Download HTML Report",
                        data=html_bytes,
                        file_name=f"credit_assessment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        use_container_width=True
                    )
                
                with col2:
                    # Create a text report alternative
                    text_report = self.generate_text_report()
                    st.download_button(
                        label="📄 Download Text Report",
                        data=text_report,
                        file_name=f"credit_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col3:
                    # CSV data export
                    csv_data = self.generate_csv_data()
                    st.download_button(
                        label="📊 Download CSV Data",
                        data=csv_data,
                        file_name=f"credit_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                st.info("""
                **📋 Report Includes:**
                - Complete applicant information
                - Credit assessment results from all AI models
                - Risk level analysis and recommendations
                - Detailed model performance
                - Timestamp and platform details
                """)
        else:
            st.warning("Please complete credit assessment first to generate reports.")

    def generate_text_report(self):
        """Generate text format report"""
        if not self.current_prediction:
            return ""
            
        features = self.current_prediction['features']
        final_score = self.current_prediction['final_score']
        risk_level = self.current_prediction['risk_level']
        
        report = f"""
AI CREDIT SCORING 2.0 - CREDIT ASSESSMENT REPORT
================================================

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

APPLICANT INFORMATION:
---------------------
Credit Score: {features[0]}
Annual Income: ₹{features[1]:,}
Employment Length: {features[2]} years
DTI Ratio: {features[3]:.1%}
Credit Utilization: {features[4]:.1%}
Total Accounts: {features[5]}
Derogatory Marks: {features[6]}
Savings Balance: ₹{features[7]:,}

ASSESSMENT RESULTS:
------------------
Final Credit Score: {final_score:.0f}
Risk Level: {risk_level}
Bureau Score: {300 + self.current_prediction['bureau']['ensemble'] * 550:.0f}
AI 2.0 Score: {300 + self.current_prediction['ai_scoring']['ensemble'] * 550:.0f}
Default Probability: {self.current_prediction['bureau']['ensemble']:.1%}

RECOMMENDATIONS:
---------------
"""
        
        if risk_level == "LOW RISK":
            report += "✅ APPROVAL RECOMMENDED\n- Credit Limit: ₹8,00,000\n- Interest Rate: 10.5% p.a.\n- Loan Term: 36 months"
        elif risk_level == "MEDIUM RISK":
            report += "⚠️ CONDITIONAL APPROVAL\n- Credit Limit: ₹4,00,000\n- Interest Rate: 14.5% p.a.\n- Loan Term: 24 months"
        else:
            report += "🔴 FURTHER REVIEW REQUIRED\n- Credit Limit: ₹1,50,000\n- Interest Rate: 18.5% p.a.\n- Loan Term: 12 months"

        report += "\n\nMODEL SCORES:\n------------\n"
        
        # Bureau models
        for model, score in self.current_prediction['bureau'].items():
            if model != 'ensemble':
                report += f"Bureau {model.upper()}: {300 + score * 550:.0f}\n"
        
        # AI models
        for model, score in self.current_prediction['ai_scoring'].items():
            if model != 'ensemble':
                report += f"AI 2.0 {model.upper()}: {300 + score * 550:.0f}\n"

        report += f"\nReport generated by AI Credit Scoring 2.0 Enterprise Platform"
        
        return report

    def generate_csv_data(self):
        """Generate CSV data export"""
        if not self.current_prediction:
            return ""
            
        features = self.current_prediction['features']
        
        # Create DataFrame
        data = {
            'Parameter': [
                'Credit_Score', 'Annual_Income', 'Employment_Length', 'DTI_Ratio',
                'Credit_Utilization', 'Total_Accounts', 'Derogatory_Marks', 'Savings_Balance',
                'Final_Score', 'Risk_Level', 'Bureau_Score', 'AI_Score', 'Default_Probability'
            ],
            'Value': [
                features[0], features[1], features[2], features[3],
                features[4], features[5], features[6], features[7],
                self.current_prediction['final_score'],
                self.current_prediction['risk_level'],
                300 + self.current_prediction['bureau']['ensemble'] * 550,
                300 + self.current_prediction['ai_scoring']['ensemble'] * 550,
                self.current_prediction['bureau']['ensemble']
            ]
        }
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)

    def run(self):
        """Run the complete application"""
        self.setup_application()
        
        # Load models
        if not self.loaded:
            with st.spinner("🚀 Initializing AI Credit Scoring System..."):
                if not self.load_all_models():
                    st.error("❌ Failed to initialize system. Please check configuration.")
                    return
        
        # Render header
        self.render_enterprise_header()
        
        # Step 1: Credit Scoring Form
        assessment_completed = self.render_credit_scoring_form()
        
        if assessment_completed:
            # Step 2: Scoring Results
            self.render_scoring_results()
            
            # Step 3: Model Analytics
            self.render_model_analytics()
            
            # Step 4: Portfolio Analytics
            self.render_portfolio_analytics()
            
            # Step 5: PDF Report (HTML alternative)
            self.render_pdf_report_section()
        
        else:
            # Show instructions when no assessment done
            st.markdown("""
            <div class="enterprise-card">
                <h3>🎯 How to Use This Platform</h3>
                <ol>
                    <li><strong>Fill the Credit Application Form</strong> above with applicant details</li>
                    <li><strong>Click "Analyze Credit Risk"</strong> to get AI-powered assessment</li>
                    <li><strong>Review Comprehensive Analytics</strong> including model performance and risk analysis</li>
                    <li><strong>Download Detailed Reports</strong> in multiple formats (HTML, Text, CSV)</li>
                </ol>
                
                <h4>📊 What You'll Get:</h4>
                <ul>
                    <li>✅ Real-time credit scoring with multiple AI models</li>
                    <li>✅ Detailed risk assessment and recommendations</li>
                    <li>✅ Comprehensive model analytics and comparisons</li>
                    <li>✅ Portfolio-level risk insights</li>
                    <li>✅ Professional reports in multiple formats</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    app = AICreditScoring2()
    app.run()
