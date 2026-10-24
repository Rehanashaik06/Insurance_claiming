# Insurance Customer Claim Likelihood Scoring System

## Overview
This system predicts the likelihood of insurance customers filing future claims using machine learning, helping insurance companies make better underwriting decisions and risk assessments.

## Implementation Results

### 1. Dataset Creation ✅
- Generated 10,000 synthetic customer records
- Features: age, gender, income, policy type, premium amount, policy duration, previous claims, credit score, location risk
- Target: Binary classification (will claim / won't claim)

### 2. Data Preprocessing ✅
- Encoded categorical variables (gender, policy type, location risk)
- Created engineered features:
  - income_per_premium ratio
  - claims_per_year rate
  - high_risk_age indicator
- Standardized numerical features

### 3. Model Performance ✅
- **Algorithm**: Random Forest Classifier
- **ROC AUC Score**: 0.691 (Good predictive power)
- **Accuracy**: 67%
- **Precision**: 61% for claim prediction
- **Recall**: 43% for claim prediction

### 4. Key Risk Factors (Feature Importance) ✅
1. **Credit Score** (15.3%) - Most important predictor
2. **Premium Amount** (13.5%)
3. **Income** (13.3%)
4. **Income-to-Premium Ratio** (13.2%)
5. **Age** (11.7%)

### 5. Risk Segmentation Results ✅

| Risk Segment | Avg Age | Avg Income | Previous Claims | Credit Score | Claim Rate |
|--------------|---------|------------|-----------------|--------------|------------|
| Low Risk     | 47.1    | $49,607    | 0.28           | 593          | 3%         |
| Medium Risk  | 50.3    | $50,467    | 0.65           | 555          | 28%        |
| High Risk    | 48.7    | $49,979    | 0.51           | 564          | 85%        |
| Very High Risk| 52.3   | $49,588    | 0.98           | 541          | 99%        |

## Business Insights & Recommendations

### High-Risk Customer Profile
- Age: 52+ years
- Credit Score: <550
- Previous Claims: 1+ claims
- Strong correlation between credit score and claim likelihood

### Underwriting Recommendations
1. **Premium Adjustments**: Increase premiums by 25-40% for very high-risk customers
2. **Enhanced Screening**: Customers with credit score <600 require additional review
3. **Risk Surcharges**: Apply surcharges for customers with >2 previous claims
4. **Age + Location Risk**: Enhanced screening for customers >65 in high-risk locations

### Business Impact
- **22.5%** of customers are in very high-risk category
- **98.8%** claim rate for very high-risk customers vs **3%** for low-risk
- Potential to reduce claim losses through better risk pricing

## Files Generated
1. `insurance_claim_scoring.py` - Main implementation
2. `risk_analysis.py` - Detailed analysis and insights
3. `customer_risk_scores.csv` - Customer data with risk scores
4. `feature_importance.png` - Feature importance visualization
5. `confusion_matrix.png` - Model performance visualization
6. `risk_analysis.png` - Risk segment analysis charts

## Usage Example
```python
# Calculate risk score for new customer
risk_score = calculate_risk_score(model, scaler, customer_data, feature_cols)
print(f"Customer Risk Score: {risk_score}/100")
```

## Next Steps
1. Deploy model as API endpoint
2. Integrate with underwriting systems
3. Monitor model performance over time
4. Retrain with real claim data
5. A/B test premium adjustments

## Technical Requirements
- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, seaborn
- Install: `pip install -r requirements.txt`

## Model Limitations
- Based on synthetic data - real-world performance may vary
- Requires periodic retraining with actual claim outcomes
- Consider regulatory compliance for insurance scoring