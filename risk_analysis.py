import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from insurance_claim_scoring import create_synthetic_dataset, preprocess_data, train_model

def analyze_risk_segments(df, model, scaler, feature_cols):
    """Analyze different customer segments and their risk profiles"""
    
    # Calculate risk scores for all customers
    X_all = df[feature_cols]
    X_all_scaled = scaler.transform(X_all)
    df['risk_score'] = model.predict_proba(X_all_scaled)[:, 1] * 100
    
    # Risk segments
    df['risk_segment'] = pd.cut(df['risk_score'], 
                               bins=[0, 25, 50, 75, 100], 
                               labels=['Low', 'Medium', 'High', 'Very High'])
    
    print("Risk Segment Analysis:")
    print("=" * 30)
    segment_analysis = df.groupby('risk_segment').agg({
        'age': 'mean',
        'income': 'mean',
        'previous_claims': 'mean',
        'credit_score': 'mean',
        'will_claim': 'mean'
    }).round(2)
    
    print(segment_analysis)
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Risk score distribution
    axes[0,0].hist(df['risk_score'], bins=30, alpha=0.7)
    axes[0,0].set_title('Risk Score Distribution')
    axes[0,0].set_xlabel('Risk Score')
    
    # Age vs Risk Score
    axes[0,1].scatter(df['age'], df['risk_score'], alpha=0.5)
    axes[0,1].set_title('Age vs Risk Score')
    axes[0,1].set_xlabel('Age')
    axes[0,1].set_ylabel('Risk Score')
    
    # Policy Type Risk
    policy_risk = df.groupby('policy_type')['risk_score'].mean().sort_values(ascending=False)
    axes[1,0].bar(policy_risk.index, policy_risk.values)
    axes[1,0].set_title('Average Risk Score by Policy Type')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Location Risk
    location_risk = df.groupby('location_risk')['risk_score'].mean()
    axes[1,1].bar(location_risk.index, location_risk.values)
    axes[1,1].set_title('Average Risk Score by Location Risk')
    
    plt.tight_layout()
    plt.savefig('d:/Assignment/risk_analysis.png')
    plt.show()
    
    return df

def generate_business_insights(df):
    """Generate actionable business insights"""
    
    print("\nBusiness Insights & Recommendations:")
    print("=" * 40)
    
    # High-risk characteristics
    high_risk = df[df['risk_segment'] == 'Very High']
    
    print(f"1. High-Risk Customer Profile:")
    print(f"   - Average Age: {high_risk['age'].mean():.1f} years")
    print(f"   - Average Income: ${high_risk['income'].mean():,.0f}")
    print(f"   - Average Previous Claims: {high_risk['previous_claims'].mean():.1f}")
    print(f"   - Average Credit Score: {high_risk['credit_score'].mean():.0f}")
    
    # Premium recommendations
    avg_premium = df['premium_amount'].mean()
    high_risk_premium = high_risk['premium_amount'].mean()
    
    print(f"\n2. Premium Adjustment Recommendations:")
    print(f"   - Current average premium: ${avg_premium:.0f}")
    print(f"   - High-risk customer premium: ${high_risk_premium:.0f}")
    print(f"   - Suggested premium increase for high-risk: 25-40%")
    
    # Underwriting rules
    print(f"\n3. Suggested Underwriting Rules:")
    print(f"   - Customers with >2 previous claims: Require additional review")
    print(f"   - Credit score <600: Apply risk surcharge")
    print(f"   - Age >65 + High location risk: Enhanced screening")
    
    # ROI Analysis
    total_customers = len(df)
    high_risk_customers = len(high_risk)
    claim_rate = df['will_claim'].mean()
    
    print(f"\n4. Business Impact:")
    print(f"   - Total customers: {total_customers:,}")
    print(f"   - High-risk customers: {high_risk_customers:,} ({high_risk_customers/total_customers*100:.1f}%)")
    print(f"   - Overall claim rate: {claim_rate*100:.1f}%")
    print(f"   - High-risk claim rate: {high_risk['will_claim'].mean()*100:.1f}%")

if __name__ == "__main__":
    # Load and prepare data
    df = create_synthetic_dataset()
    df = preprocess_data(df)
    model, scaler, _, _, feature_cols = train_model(df)
    
    # Perform detailed analysis
    df_with_scores = analyze_risk_segments(df, model, scaler, feature_cols)
    generate_business_insights(df_with_scores)
    
    # Save results
    df_with_scores.to_csv('d:/Assignment/customer_risk_scores.csv', index=False)
    print(f"\nResults saved to customer_risk_scores.csv")