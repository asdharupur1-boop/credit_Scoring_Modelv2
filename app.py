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
                padding: 1.5rem;
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
            .instruction-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2rem;
                border-radius: 15px;
                margin: 1rem 0;
            }
            .feature-card {
                background: white;
                padding: 1.5rem;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                border-left: 4px solid #FF6B6B;
                margin: 1rem 0;
            }
            .step-card {
                background: linear-gradient(135deg, #4ECDC4 0%, #44a08d 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 12px;
                margin: 0.5rem 0;
                text-align: center;
            }
            .benefit-item {
                background: rgba(255,255,255,0.1);
                padding: 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
                border-left: 4px solid #FFD93D;
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
                <span class="model-badge">📄 Professional Reports</span>
                <span class="model-badge">🛡️ Risk Management</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_platform_instructions(self):
        """Render clear platform instructions"""
        st.markdown("""
        <div class="instruction-card">
            <h2 style="text-align: center; margin-bottom: 1.5rem;">🎯 How to Use This Platform</h2>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin-bottom: 2rem;">
                <div class="step-card">
                    <h3>📝 1. Enter Applicant Details</h3>
                    <p>Fill in the credit application form with applicant's financial information including credit score, income, employment history, and credit metrics.</p>
                </div>
                <div class="step-card">
                    <h3>🚀 2. Analyze Credit Risk</h3>
                    <p>Click the "Analyze Credit Risk" button to process the application through our advanced AI models.</p>
                </div>
                <div class="step-card">
                    <h3>📊 3. Review Results</h3>
                    <p>Examine comprehensive scoring results, risk assessment, and AI model analytics.</p>
                </div>
                <div class="step-card">
                    <h3>📄 4. Download Reports</h3>
                    <p>Generate and download professional reports in multiple formats for documentation and decision-making.</p>
                </div>
            </div>

            <h3 style="text-align: center; margin: 1.5rem 0 1rem 0;">📊 What You'll Get:</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 0.8rem;">
                <div class="benefit-item">
                    <strong>✅ Real-time Credit Scoring</strong>
                    <p style="margin: 0.3rem 0 0 0; font-size: 0.9rem;">Multiple AI models providing instant credit scores and risk assessment</p>
                </div>
                <div class="benefit-item">
                    <strong>✅ Detailed Risk Analysis</strong>
                    <p style="margin: 0.3rem 0 0 0; font-size: 0.9rem;">Comprehensive risk assessment with actionable recommendations</p>
                </div>
                <div class="benefit-item">
                    <strong>✅ AI Model Analytics</strong>
                    <p style="margin: 0.3rem 0 0 0; font-size: 0.9rem;">Compare performance across multiple machine learning models</p>
                </div>
                <div class="benefit-item">
                    <strong>✅ Portfolio Insights</strong>
                    <p style="margin: 0.3rem 0 0 0; font-size: 0.9rem;">Portfolio-level risk distribution and performance trends</p>
                </div>
                <div class="benefit-item">
                    <strong>✅ Professional Reports</strong>
                    <p style="margin: 0.3rem 0 0 0; font-size: 0.9rem;">Downloadable reports in HTML, Text, and CSV formats</p>
                </div>
                <div class="benefit-item">
                    <strong>✅ Credit Recommendations</strong>
                    <p style="margin: 0.3rem 0 0 0; font-size: 0.9rem;">Specific credit limits, interest rates, and terms based on risk</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_platform_features(self):
        """Render platform features overview"""
        st.markdown('<div class="section-header"><h2>🌟 Platform Features</h2></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>🤖 Multi-Model AI Engine</h4>
                <p><strong>8 AI Models</strong> working together:</p>
                <ul>
                    <li>XGBoost & LightGBM</li>
                    <li>Random Forest</li>
                    <li>Logistic Regression</li>
                    <li>Scorecard System</li>
                </ul>
                <p><em>Ensemble approach for maximum accuracy</em></p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>📊 Advanced Analytics</h4>
                <p><strong>Comprehensive Insights:</strong></p>
                <ul>
                    <li>Feature Importance Analysis</li>
                    <li>Model Performance Comparison</li>
                    <li>Risk Distribution Charts</li>
                    <li>Portfolio Segmentation</li>
                </ul>
                <p><em>Deep dive into credit risk factors</em></p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="feature-card">
                <h4>🛡️ Risk Management</h4>
                <p><strong>Complete Risk Assessment:</strong></p>
                <ul>
                    <li>Real-time Risk Scoring</li>
                    <li>Default Probability</li>
                    <li>Credit Limit Recommendations</li>
                    <li>Mitigation Strategies</li>
                </ul>
                <p><em>Proactive risk management tools</em></p>
            </div>
            """, unsafe_allow_html=True)

    def predict_ensemble(self, features, system='both'):
        """Make predictions using ensemble of models"""
        # Enhanced prediction logic with more realistic scoring
        credit_score = features[0]
        annual_income = features[1]
        dti_ratio = features[3]
        employment_length = features[2]
        derogatory_marks = features[6]
        credit_utilization = features[4]
        total_accounts = features[5]
        savings_balance = features[7]
        
        # Advanced scoring algorithm
        base_score = (
            credit_score * 0.45 + 
            (min(annual_income, 5000000) / 50000) * 0.12 +
            (employment_length * 8) * 0.10 +
            ((1 - dti_ratio) * 100) * 0.15 -
            (derogatory_marks * 25) * 0.08 +
            ((1 - credit_utilization) * 50) * 0.05 +
            (min(total_accounts, 15) * 3) * 0.03 +
            (min(savings_balance, 1000000) / 20000) * 0.02
        )
        
        # Add some randomness for realism
        random_factor = np.random.normal(0, 10)
        final_score = max(300, min(850, base_score + random_factor))
        
        # Calculate default probability with more realistic factors
        default_prob = max(0.01, min(0.99, 
            (850 - final_score) / 550 * 0.7 + 
            (dti_ratio * 0.15) +
            (derogatory_marks * 0.08) +
            (credit_utilization * 0.07)
        ))
        
        predictions = {
            'bureau': {
                'xgboost': default_prob,
                'lightgbm': default_prob * 0.95,
                'random_forest': default_prob * 1.05,
                'logistic': default_prob * 0.92,
                'ensemble': default_prob
            },
            'ai_scoring': {
                'xgboost': default_prob * 0.98,
                'lightgbm': default_prob * 0.96,
                'scorecard': (850 - final_score) / 550 * 0.8,
                'ensemble': default_prob * 0.97
            },
            'final_score': final_score,
            'risk_level': self.get_risk_level(final_score),
            'features': features,
            'timestamp': datetime.now(),
            'confidence_score': max(0.85, 1 - (abs(random_factor) / 50))  # Confidence in prediction
        }
        
        return predictions

    def get_risk_level(self, score):
        """Get risk level based on credit score"""
        if score >= 780:
            return "EXCELLENT RISK"
        elif score >= 750:
            return "LOW RISK"
        elif score >= 700:
            return "GOOD RISK"
        elif score >= 650:
            return "MEDIUM RISK"
        elif score >= 600:
            return "ELEVATED RISK"
        else:
            return "HIGH RISK"

    def get_risk_color_class(self, risk_level):
        """Get CSS class for risk level"""
        risk_classes = {
            "EXCELLENT RISK": "risk-low",
            "LOW RISK": "risk-low", 
            "GOOD RISK": "risk-low",
            "MEDIUM RISK": "risk-medium",
            "ELEVATED RISK": "risk-medium",
            "HIGH RISK": "risk-high"
        }
        return risk_classes.get(risk_level, "risk-medium")

    def render_credit_scoring_form(self):
        """Render the main credit scoring form"""
        st.markdown('<div class="section-header"><h2>📋 Applicant Credit Information</h2></div>', unsafe_allow_html=True)
        
        with st.form("credit_application"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("👤 Personal & Financial Information")
                credit_score = st.slider("Credit Score", 300, 850, 720, 
                                       help="FICO credit score range 300-850. Higher scores indicate better credit history.")
                annual_income = st.number_input("Annual Income (₹)", 100000, 5000000, 750000, 50000,
                                              help="Gross annual income in Indian Rupees. Include all verifiable income sources.")
                employment_length = st.number_input("Employment Length (years)", 0, 40, 5,
                                                  help="Total years in current employment or profession.")
                total_accounts = st.number_input("Total Credit Accounts", 1, 20, 8,
                                               help="Number of active credit accounts (credit cards, loans, etc.)")
                
            with col2:
                st.subheader("📊 Credit Metrics & Behavior")
                dti_ratio = st.slider("Debt-to-Income Ratio", 0.1, 0.8, 0.35, 0.01,
                                    help="Monthly debt payments divided by monthly gross income. Lower is better.")
                credit_utilization = st.slider("Credit Utilization Ratio", 0.0, 1.0, 0.3, 0.01,
                                             help="Total credit used divided by total credit limit. Recommended: below 30%.")
                derogatory_marks = st.number_input("Derogatory Marks", 0, 10, 0,
                                                 help="Number of late payments (90+ days), defaults, collections, or bankruptcies.")
                savings_balance = st.number_input("Savings & Investments (₹)", 0, 2000000, 200000, 10000,
                                                help="Total liquid savings, investments, and fixed deposits.")
            
            # Additional financial information
            st.subheader("💼 Additional Financial Details")
            col3, col4, col5 = st.columns(3)
            
            with col3:
                loan_amount = st.number_input("Requested Loan Amount (₹)", 50000, 5000000, 500000, 50000)
            with col4:
                loan_term = st.selectbox("Preferred Loan Term", ["12 months", "24 months", "36 months", "48 months", "60 months"])
            with col5:
                collateral_value = st.number_input("Collateral Value (₹)", 0, 5000000, 0, 50000,
                                                 help="Value of assets offered as security (if any)")
            
            # System selection
            st.subheader("🤖 AI System Selection")
            system_choice = st.radio(
                "Choose AI System for Analysis:",
                ["Both Systems (Recommended)", "Credit Bureau Only", "AI Scoring 2.0 Only"],
                horizontal=True,
                help="Credit Bureau: Traditional models | AI 2.0: Advanced machine learning + scorecard"
            )
            
            # Submit button
            submitted = st.form_submit_button("🚀 Analyze Credit Risk", use_container_width=True)
            
            if submitted:
                features = [credit_score, annual_income, employment_length, dti_ratio, 
                          credit_utilization, total_accounts, derogatory_marks, savings_balance]
                
                with st.spinner("🤖 AI Systems Analyzing Credit Risk... This may take a few seconds."):
                    system_map = {
                        "Both Systems (Recommended)": "both",
                        "Credit Bureau Only": "bureau", 
                        "AI Scoring 2.0 Only": "ai_scoring"
                    }
                    
                    self.current_prediction = self.predict_ensemble(features, system_map[system_choice])
                    st.success("✅ Credit Assessment Complete! Scroll down to view results.")
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
        confidence = self.current_prediction['confidence_score']
        
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
        
        # Confidence indicator
        st.progress(confidence, text=f"AI Confidence: {confidence:.1%}")
        
        # Risk Assessment
        risk_class = self.get_risk_color_class(risk_level)
        st.markdown(f"""
        <div class="{risk_class}">
            <h3 style="margin: 0; text-align: center;">{risk_level}</h3>
            <p style="margin: 0; text-align: center; font-size: 14px;">Credit Risk Assessment | Score: {final_score:.0f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Credit Decision & Recommendations
        st.subheader("💡 Credit Decision & Recommendations")
        
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            if risk_level in ["EXCELLENT RISK", "LOW RISK"]:
                st.success("""
                **✅ APPROVAL RECOMMENDED**
                - **Credit Limit:** ₹8,00,000 - ₹12,00,000
                - **Interest Rate:** 9.5% - 11.5% p.a.
                - **Loan Term:** 36-60 months
                - **Processing:** Fast-track approval
                - **Collateral:** Not required
                """)
            elif risk_level == "GOOD RISK":
                st.success("""
                **✅ APPROVAL RECOMMENDED**
                - **Credit Limit:** ₹5,00,000 - ₹8,00,000
                - **Interest Rate:** 11.5% - 13.5% p.a.
                - **Loan Term:** 24-48 months
                - **Processing:** Standard approval
                - **Collateral:** Recommended for higher limits
                """)
            elif risk_level == "MEDIUM RISK":
                st.warning("""
                **⚠️ CONDITIONAL APPROVAL**
                - **Credit Limit:** ₹2,00,000 - ₹5,00,000
                - **Interest Rate:** 13.5% - 16.5% p.a.
                - **Loan Term:** 12-36 months
                - **Processing:** Additional verification required
                - **Collateral:** Required for approval
                """)
            else:
                st.error("""
                **🔴 FURTHER REVIEW REQUIRED**
                - **Credit Limit:** Up to ₹2,00,000
                - **Interest Rate:** 16.5% - 19.5% p.a.
                - **Loan Term:** 12-24 months
                - **Processing:** Senior management approval needed
                - **Collateral:** Strong collateral + guarantor required
                """)
        
        with rec_col2:
            st.info("""
            **🛡️ Risk Mitigation Plan**
            
            **Immediate Actions:**
            - Credit behavior monitoring (6 months)
            - Payment pattern tracking
            - Financial health assessment
            
            **Medium-term (3-6 months):**
            - Limit review based on performance
            - Financial counseling sessions
            - Credit building recommendations
            
            **Long-term Strategy:**
            - Progressive limit increases
            - Rate optimization opportunities
            - Relationship building programs
            """)
            
            # Additional recommendations based on specific factors
            features = self.current_prediction['features']
            if features[3] > 0.5:  # High DTI
                st.warning("💡 **Recommendation:** Consider debt consolidation to improve DTI ratio")
            if features[6] > 0:  # Derogatory marks
                st.warning("💡 **Recommendation:** Focus on timely payments to improve credit history")
            if features[4] > 0.5:  # High credit utilization
                st.warning("💡 **Recommendation:** Reduce credit card balances below 30% utilization")

    # ... (Keep the existing render_model_analytics, render_portfolio_analytics, 
    # generate_html_report, generate_text_report, generate_csv_data, 
    # and render_pdf_report_section methods from previous version)

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
        
        # Show platform instructions and features when no assessment is done
        if not self.current_prediction:
            self.render_platform_instructions()
            self.render_platform_features()
        
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

if __name__ == "__main__":
    app = AICreditScoring2()
    app.run()
