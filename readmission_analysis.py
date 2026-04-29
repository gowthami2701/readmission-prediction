"""
30-Day Hospital Readmission Prediction
CMS HRRP Compliance Analytics
Author: Gowthami Vasamsetti
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, classification_report, 
                              confusion_matrix, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── DATA GENERATION ───────────────────────────────────────────────────────────

def generate_patient_data(n: int = 10000) -> pd.DataFrame:
    """Generate realistic hospital patient data for readmission modeling."""
    
    print(f"Generating {n} patient records...")
    
    # Demographics
    age = np.random.normal(65, 15, n).clip(18, 99).astype(int)
    gender = np.random.choice(['M', 'F'], n, p=[0.48, 0.52])
    race = np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], 
                             n, p=[0.60, 0.18, 0.14, 0.05, 0.03])
    
    # Insurance
    insurance = np.where(age >= 65, 
                         np.random.choice(['Medicare FFS', 'Medicare Advantage'], n, p=[0.6, 0.4]),
                         np.random.choice(['Medicaid', 'Commercial', 'Self-Pay'], n, p=[0.35, 0.55, 0.10]))
    
    # Primary diagnosis (CMS HRRP conditions weighted higher)
    dx_choices = ['Heart Failure', 'AMI', 'Pneumonia', 'COPD', 
                  'Hip/Knee Replacement', 'CABG', 'Sepsis', 'Other']
    dx_weights = [0.18, 0.12, 0.15, 0.13, 0.10, 0.07, 0.10, 0.15]
    primary_dx = np.random.choice(dx_choices, n, p=dx_weights)
    
    # Clinical features
    n_chronic = np.random.poisson(2.5, n).clip(0, 10)
    los = np.random.lognormal(1.8, 0.6, n).clip(1, 60).astype(int)  # Length of stay
    prior_admissions_30d = np.random.choice([0,1,2,3], n, p=[0.72, 0.18, 0.07, 0.03])
    prior_admissions_6m = prior_admissions_30d + np.random.poisson(0.8, n)
    
    # Lab values
    sodium = np.random.normal(138, 4, n).clip(120, 155)
    bun = np.random.lognormal(3.0, 0.4, n).clip(7, 150)
    creatinine = np.random.lognormal(0.2, 0.5, n).clip(0.5, 15)
    hematocrit = np.random.normal(38, 6, n).clip(20, 55)
    
    # Process/care quality features
    discharge_disposition = np.random.choice(
        ['Home', 'Home with Services', 'SNF', 'Rehab', 'AMA'], 
        n, p=[0.45, 0.25, 0.18, 0.10, 0.02])
    followup_appt_scheduled = np.random.choice([0,1], n, p=[0.28, 0.72])
    discharge_instructions_complete = np.random.choice([0,1], n, p=[0.15, 0.85])
    weekend_discharge = np.random.choice([0,1], n, p=[0.71, 0.29])
    
    # ── READMISSION OUTCOME (clinically realistic) ────────────────────────────
    # Base risk
    readmit_prob = 0.08 * np.ones(n)
    
    # Age effect
    readmit_prob += np.where(age > 75, 0.06, np.where(age > 65, 0.03, 0))
    
    # Diagnosis effect (HRRP conditions have higher readmit rates)
    hrrp_conditions = ['Heart Failure', 'AMI', 'Pneumonia', 'COPD', 'CABG']
    readmit_prob += np.where(np.isin(primary_dx, hrrp_conditions), 0.08, 0)
    readmit_prob += np.where(primary_dx == 'Heart Failure', 0.05, 0)
    
    # Prior admissions (strongest predictor)
    readmit_prob += prior_admissions_30d * 0.12
    readmit_prob += prior_admissions_6m * 0.02
    
    # Chronic conditions
    readmit_prob += n_chronic * 0.02
    
    # LOS effect
    readmit_prob += np.where(los > 7, 0.05, 0)
    
    # Insurance (Medicaid/Self-pay = higher risk)
    readmit_prob += np.where(np.isin(insurance, ['Medicaid', 'Self-Pay']), 0.04, 0)
    
    # Protective factors
    readmit_prob -= np.where(followup_appt_scheduled == 1, 0.04, 0)
    readmit_prob -= np.where(discharge_instructions_complete == 1, 0.03, 0)
    readmit_prob -= np.where(discharge_disposition == 'SNF', 0.02, 0)
    
    # Add noise and clip
    readmit_prob += np.random.normal(0, 0.03, n)
    readmit_prob = readmit_prob.clip(0.01, 0.95)
    
    readmitted = (np.random.random(n) < readmit_prob).astype(int)
    
    df = pd.DataFrame({
        'patient_id':                   [f'PT{i:06d}' for i in range(n)],
        'age':                          age,
        'gender':                       gender,
        'race':                         race,
        'insurance_type':               insurance,
        'primary_diagnosis':            primary_dx,
        'n_chronic_conditions':         n_chronic,
        'length_of_stay':               los,
        'prior_admissions_30d':         prior_admissions_30d,
        'prior_admissions_6m':          prior_admissions_6m,
        'sodium_at_discharge':          sodium.round(1),
        'bun_at_discharge':             bun.round(1),
        'creatinine_at_discharge':      creatinine.round(2),
        'hematocrit_at_discharge':      hematocrit.round(1),
        'discharge_disposition':        discharge_disposition,
        'followup_appt_scheduled':      followup_appt_scheduled,
        'discharge_instructions_complete': discharge_instructions_complete,
        'weekend_discharge':            weekend_discharge,
        'readmitted_30day':             readmitted,
    })
    
    print(f"✅ Generated {n} patients | Readmission Rate: {readmitted.mean():.1%}")
    return df


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create clinically meaningful derived features."""
    df = df.copy()
    
    # Elixhauser-inspired comorbidity risk score
    df['high_comorbidity'] = (df['n_chronic_conditions'] >= 4).astype(int)
    
    # Renal function flag (BUN:Creatinine ratio)
    df['bun_cr_ratio'] = df['bun_at_discharge'] / df['creatinine_at_discharge'].clip(lower=0.1)
    df['renal_risk'] = (df['bun_cr_ratio'] > 20).astype(int)
    
    # Anemia flag
    df['anemia'] = (df['hematocrit_at_discharge'] < 33).astype(int)
    
    # Hyponatremia flag (low sodium = bad outcomes)
    df['hyponatremia'] = (df['sodium_at_discharge'] < 135).astype(int)
    
    # Age risk groups
    df['age_group'] = pd.cut(df['age'], bins=[0,45,65,75,100], 
                              labels=['<45','45-65','65-75','>75'])
    
    # HRRP condition flag
    hrrp = ['Heart Failure','AMI','Pneumonia','COPD','Hip/Knee Replacement','CABG']
    df['hrrp_condition'] = df['primary_diagnosis'].isin(hrrp).astype(int)
    
    # Readmission history risk
    df['prior_admit_risk'] = (df['prior_admissions_30d'] > 0).astype(int)
    
    # Care quality composite score
    df['care_quality_score'] = (
        df['followup_appt_scheduled'] + 
        df['discharge_instructions_complete'] + 
        (1 - df['weekend_discharge'])
    )
    
    # Encode categoricals
    le = LabelEncoder()
    for col in ['gender', 'race', 'insurance_type', 'primary_diagnosis', 
                'discharge_disposition', 'age_group']:
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
    
    return df


# ── MODELING PIPELINE ─────────────────────────────────────────────────────────

def build_and_evaluate_models(df: pd.DataFrame):
    """Train and evaluate readmission prediction models."""
    
    feature_cols = [
        'age', 'gender_enc', 'race_enc', 'insurance_type_enc',
        'primary_diagnosis_enc', 'n_chronic_conditions', 'length_of_stay',
        'prior_admissions_30d', 'prior_admissions_6m',
        'sodium_at_discharge', 'bun_at_discharge', 'creatinine_at_discharge',
        'hematocrit_at_discharge', 'discharge_disposition_enc',
        'followup_appt_scheduled', 'discharge_instructions_complete',
        'weekend_discharge', 'high_comorbidity', 'bun_cr_ratio',
        'renal_risk', 'anemia', 'hyponatremia', 'hrrp_condition',
        'prior_admit_risk', 'care_quality_score', 'age_group_enc'
    ]
    
    X = df[feature_cols]
    y = df['readmitted_30day']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\n📊 Dataset Split:")
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"   Readmission Rate (train): {y_train.mean():.1%}")
    print(f"   Readmission Rate (test):  {y_test.mean():.1%}")
    
    models = {
        'Logistic Regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(random_state=42, max_iter=1000))
        ]),
        'Random Forest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestClassifier(n_estimators=200, max_depth=8, 
                                              random_state=42, n_jobs=-1))
        ]),
    }
    
    # Try XGBoost if available
    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                     eval_metric='logloss', random_state=42, 
                                     use_label_encoder=False))
        ])
    except ImportError:
        print("   (XGBoost not available, skipping)")
    
    print("\n🤖 MODEL COMPARISON")
    print("=" * 65)
    print(f"{'Model':<25} {'AUC-ROC':>8} {'CV AUC':>8} {'F1':>8}")
    print("-" * 65)
    
    best_model = None
    best_auc = 0
    results = {}
    
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred_proba = pipe.predict_proba(X_test)[:, 1]
        y_pred = pipe.predict(X_test)
        
        auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = report['1']['f1-score']
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, 
                                  scoring='roc_auc', n_jobs=-1).mean()
        
        print(f"{name:<25} {auc:>8.4f} {cv_auc:>8.4f} {f1:>8.4f}")
        results[name] = {'model': pipe, 'auc': auc, 'proba': y_pred_proba}
        
        if auc > best_auc:
            best_auc = auc
            best_model = name
    
    print("-" * 65)
    print(f"\n✅ Best Model: {best_model} (AUC: {best_auc:.4f})")
    
    # Risk stratification using best model
    best_pipe = results[best_model]['model']
    df_test = X_test.copy()
    df_test['readmit_probability'] = results[best_model]['proba']
    df_test['risk_tier'] = pd.cut(
        df_test['readmit_probability'],
        bins=[0, 0.10, 0.20, 0.35, 1.0],
        labels=['Low (<10%)', 'Moderate (10-20%)', 'High (20-35%)', 'Very High (>35%)']
    )
    
    print("\n🎯 RISK STRATIFICATION SUMMARY")
    print("=" * 55)
    risk_summary = df_test.groupby('risk_tier', observed=True).agg(
        Patients=('readmit_probability', 'count'),
        Avg_Risk=('readmit_probability', 'mean')
    )
    for tier, row in risk_summary.iterrows():
        intervention = {
            'Low (<10%)': 'Standard discharge',
            'Moderate (10-20%)': 'Phone follow-up at 48hrs',
            'High (20-35%)': 'Home health referral + care coordinator',
            'Very High (>35%)': 'Immediate care coordination + SNF evaluation'
        }.get(str(tier), '')
        print(f"  {str(tier):<22} {int(row['Patients']):>6} pts | Avg Risk: {row['Avg_Risk']:.1%} | {intervention}")
    
    # Feature importance
    if best_model == 'Random Forest':
        importances = best_pipe.named_steps['model'].feature_importances_
    elif best_model == 'XGBoost':
        importances = best_pipe.named_steps['model'].feature_importances_
    else:
        importances = abs(best_pipe.named_steps['model'].coef_[0])
    
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    
    print(f"\n📌 TOP 10 READMISSION RISK FACTORS ({best_model})")
    print("=" * 45)
    for feat, imp in feat_imp.head(10).items():
        bar = '█' * int(imp * 200)
        print(f"  {feat:<35} {imp:.4f} {bar}")
    
    return best_pipe, df_test


if __name__ == '__main__':
    df_raw = generate_patient_data(10000)
    df_features = engineer_features(df_raw)
    model, predictions = build_and_evaluate_models(df_features)
    
    # Save outputs
    df_raw.to_csv('data/patient_data.csv', index=False)
    predictions.to_csv('data/risk_scores.csv', index=False)
    print("\n💾 Saved patient_data.csv and risk_scores.csv")
    print("\n🏁 Project complete! Load risk_scores.csv into Tableau for dashboard.")
