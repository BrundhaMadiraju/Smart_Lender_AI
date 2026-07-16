import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

def preprocess(path='dataset/loan_data.csv'):
    df = pd.read_csv(path)

    # ── Drop Loan_ID (not a feature) ──
    df.drop(columns=['Loan_ID'], inplace=True)

    # ── Handle missing values ──
    # Categorical: fill with mode
    cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Numerical: fill with mean
    num_cols = ['LoanAmount', 'Loan_Amount_Term']
    for col in num_cols:
        df[col].fillna(df[col].mean(), inplace=True)

    print("Missing values after handling:\n", df.isnull().sum())

    # ── Encode categorical features ──
    le = LabelEncoder()
    encode_cols = ['Gender', 'Married', 'Dependents', 'Education',
                   'Self_Employed', 'Property_Area', 'Loan_Status']
    for col in encode_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # ── Features and target ──
    X = df.drop(columns=['Loan_Status'])
    y = df['Loan_Status']

    print("\nClass distribution before SMOTE:", y.value_counts().to_dict())

    # ── Balance dataset with SMOTE ──
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print("Class distribution after SMOTE:", pd.Series(y_res).value_counts().to_dict())

    # ── Scale features ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    # Save scaler for use in Flask app
    joblib.dump(scaler, 'model/scaler.pkl')

    # ── Train/test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_res, test_size=0.2, random_state=42
    )

    print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, scaler = preprocess()