import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Synthetic Insurance Dataset Generation
def create_synthetic_dataset(n_samples=10000):
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'policy_type': np.random.choice(['Auto', 'Home', 'Life', 'Health'], n_samples),
        'premium_amount': np.random.normal(1200, 400, n_samples),
        'policy_duration': np.random.randint(1, 10, n_samples),
        'previous_claims': np.random.poisson(0.5, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'location_risk': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.5, 0.3, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable with realistic correlations
    claim_prob = (
        0.1 + 
        (df['age'] > 65) * 0.15 +
        (df['previous_claims'] > 0) * 0.3 +
        (df['location_risk'] == 'High') * 0.2 +
        (df['credit_score'] < 600) * 0.15 +
        np.random.normal(0, 0.1, n_samples)
    )
    
    df['will_claim'] = (np.random.random(n_samples) < np.clip(claim_prob, 0, 1)).astype(int)
    
    return df

# 2. Data Preprocessing
def preprocess_data(df):
    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['gender', 'policy_type', 'location_risk']
    
    for col in categorical_cols:
        df[col + '_encoded'] = le.fit_transform(df[col])
    
    # Feature engineering
    df['income_per_premium'] = df['income'] / df['premium_amount']
    df['claims_per_year'] = df['previous_claims'] / df['policy_duration']
    df['high_risk_age'] = (df['age'] > 65).astype(int)
    
    return df

# 3. Risk Scoring Model
def train_model(df):
    # Select features
    feature_cols = ['age', 'income', 'premium_amount', 'policy_duration', 
                   'previous_claims', 'credit_score', 'gender_encoded', 
                   'policy_type_encoded', 'location_risk_encoded',
                   'income_per_premium', 'claims_per_year', 'high_risk_age']
    
    X = df[feature_cols]
    y = df['will_claim']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_test_scaled, y_test, feature_cols

# 4. Feature Importance Analysis
def analyze_feature_importance(model, feature_cols):
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title('Feature Importance for Claim Prediction')
    plt.tight_layout()
    plt.savefig('d:/Assignment/feature_importance.png')
    plt.show()
    
    return importance_df

# 5. Model Evaluation
def evaluate_model(model, X_test, y_test):
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"\nROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('d:/Assignment/confusion_matrix.png')
    plt.show()
    
    return y_pred_proba

# Risk Scoring Function
def calculate_risk_score(model, scaler, customer_data, feature_cols):
    """Calculate risk score (0-100) for new customers"""
    customer_scaled = scaler.transform(customer_data[feature_cols])
    risk_probability = model.predict_proba(customer_scaled)[:, 1]
    risk_score = (risk_probability * 100).round(1)
    return risk_score

# Main execution
if __name__ == "__main__":
    print("Insurance Claim Likelihood Scoring System")
    print("=" * 50)
    
    # Step 1: Create dataset
    print("1. Creating synthetic dataset...")
    df = create_synthetic_dataset()
    print(f"Dataset created with {len(df)} samples")
    
    # Step 2: Preprocess data
    print("\n2. Preprocessing data...")
    df = preprocess_data(df)
    print("Data preprocessing completed")
    
    # Step 3: Train model
    print("\n3. Training risk scoring model...")
    model, scaler, X_test, y_test, feature_cols = train_model(df)
    print("Model training completed")
    
    # Step 4: Feature importance
    print("\n4. Analyzing feature importance...")
    importance_df = analyze_feature_importance(model, feature_cols)
    print("\nTop 5 Most Important Features:")
    print(importance_df.head())
    
    # Step 5: Evaluate model
    print("\n5. Evaluating model performance...")
    risk_probabilities = evaluate_model(model, X_test, y_test)
    
    # Example risk scoring for new customers
    print("\n6. Example Risk Scoring:")
    sample_customers = df.sample(5)[feature_cols]
    risk_scores = calculate_risk_score(model, scaler, sample_customers, feature_cols)
    
    for i, score in enumerate(risk_scores):
        print(f"Customer {i+1}: Risk Score = {score}/100")
    
    print("\nSystem implementation completed!")