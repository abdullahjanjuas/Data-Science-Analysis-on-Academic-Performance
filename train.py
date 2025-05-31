import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import joblib

# === Define columns ===
numerical_cols = [
    'Higher Secondary Marks Percentage (%)',
    'NET Marks',
    'Family Size',
    'Personal Space (Sqft)'
]

ordinal_cols = [
    'Parents Education Level',
    'Family Income',
    'House Size (Sqft)',
    'Stress Level',
    'Confidence Level',
    'Study Consistency Level',
    'Participation in Extra-Curricular Activities',
    'Available Emotional Support',
    'Sleeping Level',
    'Attendance Level',
    'Access to Resources',
    'Understanding of Subject',
    'Interest in Degree'
]

categorical_cols = [
    'Gender',
    'Area',
    'Living Situation',
    'Higher Secondary Education',
    'Higher Secondary Subjects'
]

ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# === Preprocessing ===
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('ord', ordinal_encoder, ordinal_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# === Pipeline ===
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# === Fit Model ===
df = pd.read_csv("content/Final_Dataset.csv")  # Your dataset
X = df[numerical_cols + ordinal_cols + categorical_cols]
y = df['SGPA']
pipeline.fit(X, y)

# === Save pipeline ===
joblib.dump(pipeline, 'content/rf_pipeline.joblib')
