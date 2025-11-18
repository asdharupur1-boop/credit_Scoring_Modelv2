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
from fpdf import FPDF
import base64

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
        
        # Calculate base score
        base_score = credit_score * 0.6 + (annual_income / 10000) * 0.2 + (1 - dti_ratio) * 0.2
        
        # Normalize to 300-850 range
        final_score = max(300, min(850, base_score))
        
        # Calculate default probability (inverse of score)
        default_prob = max(0.01, min(0.99, (850 - final_score) / 550))
        
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
            'features': features
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
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🏦 Bureau Score", f"{300 + self.current_prediction['bureau']['ensemble'] * 550:.0f}")
        with col2:
            st.metric("🚀 AI 2.0 Score", f"{300 + self.current_prediction['ai_scoring']['ensemble'] * 550:.0f}")
        with col3:
            st.metric("📊 Final Credit Score", f"{final_score:.0f}")
        
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
                - Credit Limit: ₹8,00,000
                - Interest Rate: 10.5% p.a.
                - Terms: 36 months
                """)
            elif risk_level == "MEDIUM RISK":
                st.warning("""
                **⚠️ CONDITIONAL APPROVAL**
                - Credit Limit: ₹4,00,000  
                - Interest Rate: 14.5% p.a.
                - Terms: 24 months
                - Additional collateral recommended
                """)
            else:
                st.error("""
                **🔴 FURTHER REVIEW REQUIRED**
                - Credit Limit: ₹1,50,000
                - Interest Rate: 18.5% p.a.
                - Terms: 12 months
                - Strong collateral required
                """)
        
        with rec_col2:
            st.info("""
            **🛡️ Risk Mitigation Plan**
            - Credit monitoring for 6 months
            - Payment behavior tracking
            - Limit review after 12 months
            - Financial counseling recommended
            """)

    def render_model_analytics(self):
        """Render detailed model analytics"""
        if not self.current_prediction:
            return
            
        st.markdown('<div class="section-header"><h2>📊 AI Model Analytics</h2></div>', unsafe_allow_html=True)
        
        # Model Performance Comparison
        st.subheader("🤖 Model Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bureau Models
            bureau_data = []
            for model, score in self.current_prediction['bureau'].items():
                if model != 'ensemble':
                    bureau_data.append({
                        'Model': model.upper(),
                        'Default Probability': f"{score:.1%}",
                        'Credit Score': f"{300 + score * 550:.0f}"
                    })
            st.dataframe(pd.DataFrame(bureau_data), use_container_width=True)
        
        with col2:
            # AI Scoring Models
            ai_data = []
            for model, score in self.current_prediction['ai_scoring'].items():
                if model != 'ensemble':
                    ai_data.append({
                        'Model': model.upper(), 
                        'Default Probability': f"{score:.1%}",
                        'Credit Score': f"{300 + score * 550:.0f}"
                    })
            st.dataframe(pd.DataFrame(ai_data), use_container_width=True)
        
        # Feature Importance Visualization
        st.subheader("🎯 Feature Importance Analysis")
        
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

    def render_portfolio_analytics(self):
        """Render portfolio-level analytics"""
        st.markdown('<div class="section-header"><h2>📈 Portfolio Risk Analytics</h2></div>', unsafe_allow_html=True)
        
        # Risk Distribution
        st.subheader("📊 Risk Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            risk_data = {
                'Risk Level': ['Low Risk', 'Medium Risk', 'High Risk'],
                'Percentage': [65, 25, 10],
                'Count': [32600, 12500, 5147]
            }
            
            fig = px.pie(
                risk_data, values='Percentage', names='Risk Level',
                title="Portfolio Risk Distribution",
                color_discrete_sequence=['#4ECDC4', '#FFD93D', '#FF6B6B']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Performance trends
            trend_data = {
                'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                'Approval Rate': [78, 82, 85, 83, 86, 88],
                'Default Rate': [3.2, 2.8, 2.5, 2.3, 2.1, 1.9]
            }
            
            fig = px.line(
                trend_data, x='Month', y=['Approval Rate', 'Default Rate'],
                title="Monthly Performance Trends",
                labels={'value': 'Percentage', 'variable': 'Metric'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Customer Segmentation
        st.subheader("👥 Customer Segmentation Analysis")
        
        segment_data = {
            'Segment': ['Prime (750+)', 'Near Prime (650-749)', 'Subprime (300-649)'],
            'Customers': [25600, 18700, 5947],
            'Avg Income': ['₹12.5L', '₹8.2L', '₹5.1L'],
            'Default Rate': ['0.8%', '2.9%', '8.7%']
        }
        
        st.dataframe(pd.DataFrame(segment_data), use_container_width=True)

    def generate_pdf_report(self):
        """Generate PDF report"""
        if not self.current_prediction:
            return None
            
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'AI Credit Scoring 2.0 - Credit Assessment Report', 0, 1, 'C')
        pdf.ln(10)
        
        # Applicant Info
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Applicant Information:', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        features = self.current_prediction['features']
        pdf.cell(0, 8, f'Credit Score: {features[0]}', 0, 1)
        pdf.cell(0, 8, f'Annual Income: ₹{features[1]:,}', 0, 1)
        pdf.cell(0, 8, f'Employment Length: {features[2]} years', 0, 1)
        pdf.cell(0, 8, f'DTI Ratio: {features[3]:.1%}', 0, 1)
        pdf.cell(0, 8, f'Credit Utilization: {features[4]:.1%}', 0, 1)
        pdf.cell(0, 8, f'Total Accounts: {features[5]}', 0, 1)
        pdf.cell(0, 8, f'Derogatory Marks: {features[6]}', 0, 1)
        pdf.cell(0, 8, f'Savings Balance: ₹{features[7]:,}', 0, 1)
        
        pdf.ln(10)
        
        # Assessment Results
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Assessment Results:', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        final_score = self.current_prediction['final_score']
        risk_level = self.current_prediction['risk_level']
        
        pdf.cell(0, 8, f'Final Credit Score: {final_score:.0f}', 0, 1)
        pdf.cell(0, 8, f'Risk Level: {risk_level}', 0, 1)
        pdf.cell(0, 8, f'Bureau Score: {300 + self.current_prediction["bureau"]["ensemble"] * 550:.0f}', 0, 1)
        pdf.cell(0, 8, f'AI 2.0 Score: {300 + self.current_prediction["ai_scoring"]["ensemble"] * 550:.0f}', 0, 1)
        
        pdf.ln(10)
        
        # Recommendations
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Recommendations:', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        if risk_level == "LOW RISK":
            pdf.multi_cell(0, 8, '✅ APPROVAL RECOMMENDED\n- Credit Limit: ₹8,00,000\n- Interest Rate: 10.5% p.a.\n- Terms: 36 months')
        elif risk_level == "MEDIUM RISK":
            pdf.multi_cell(0, 8, '⚠️ CONDITIONAL APPROVAL\n- Credit Limit: ₹4,00,000\n- Interest Rate: 14.5% p.a.\n- Terms: 24 months\n- Additional collateral recommended')
        else:
            pdf.multi_cell(0, 8, '🔴 FURTHER REVIEW REQUIRED\n- Credit Limit: ₹1,50,000\n- Interest Rate: 18.5% p.a.\n- Terms: 12 months\n- Strong collateral required')
        
        # Footer
        pdf.ln(20)
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        pdf.cell(0, 10, 'AI Credit Scoring 2.0 - Enterprise Platform', 0, 1, 'C')
        
        return pdf

    def render_pdf_report_section(self):
        """Render PDF report download section"""
        st.markdown('<div class="section-header"><h2>📄 Download Comprehensive Report</h2></div>', unsafe_allow_html=True)
        
        if self.current_prediction:
            # Generate PDF
            pdf = self.generate_pdf_report()
            
            if pdf:
                # Save PDF to bytes
                pdf_output = pdf.output(dest='S').encode('latin1')
                pdf_b64 = base64.b64encode(pdf_output).decode()
                
                # Download button
                st.download_button(
                    label="📥 Download Full PDF Report",
                    data=pdf_output,
                    file_name=f"credit_assessment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
                st.info("""
                **📋 Report Includes:**
                - Applicant information
                - Credit assessment results  
                - Risk level analysis
                - AI model scores
                - Credit recommendations
                - Timestamp and platform details
                """)
        else:
            st.warning("Please complete credit assessment first to generate report.")

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
            
            # Step 5: PDF Report
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
                    <li><strong>Download PDF Report</strong> for complete documentation</li>
                </ol>
                
                <h4>📊 What You'll Get:</h4>
                <ul>
                    <li>✅ Real-time credit scoring</li>
                    <li>✅ Multi-model AI analysis</li>
                    <li>✅ Risk assessment and recommendations</li>
                    <li>✅ Portfolio-level analytics</li>
                    <li>✅ Professional PDF report</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    app = AICreditScoring2()
    app.run()
