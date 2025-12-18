import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def train():
    print("Loading data...")
    df = pd.read_csv('waec_data.csv')
    
    # 2. Cleaning: Ensure numeric types and fill missing pass_%
    cols = ['pass_eng_math', 'total_sat', 'pass_%']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Recalculate pass_% to be safe
    # This also overrides any other potential typos in pass_% if the components are correct
    # But for now we trust the components mostly
    
    # Drop bad rows
    df = df.dropna(subset=['total_sat', 'pass_%'])

    # 3. Feature Engineering
    print("Encoding features...")
    le_gender = LabelEncoder()
    df['gender_encoded'] = le_gender.fit_transform(df['gender'].astype(str))

    le_school = LabelEncoder()
    df['scholl_type_encoded'] = le_school.fit_transform(df['scholl_type'].astype(str))

    # Features
    features = ['year', 'gender_encoded', 'scholl_type_encoded', 'total_sat']
    X = df[features]
    y = df['pass_%']

    print(f"Training model with {len(df)} records...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 4. Save
    print("Saving model...")
    joblib.dump(model, 'waec_model.pkl')
    joblib.dump(le_gender, 'gender_encoder.pkl')
    joblib.dump(le_school, 'school_encoder.pkl')
    print("Done.")

if __name__ == "__main__":
    train()
