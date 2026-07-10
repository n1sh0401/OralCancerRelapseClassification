import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

root = Path.cwd()
file_path = root / 'ML TEST DATASET3.xlsx'
if not file_path.exists():
    raise FileNotFoundError(f'Dataset not found: {file_path}')

df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()
drop = ['STUDY ID', 'EARLY RECURRENCE', 'LATE RECURRENCE', 'RELAPSE (LR, LRR, SPT)']
for c in drop:
    if c in df.columns:
        df = df.drop(columns=c)

cat = df.select_dtypes(exclude=np.number).columns.tolist()
num = df.select_dtypes(include=np.number).columns.tolist()
if 'RECURRENCE (LR, LRR)' in cat:
    cat.remove('RECURRENCE (LR, LRR)')

preprocessor = ColumnTransformer(
    transformers=[
        ('nums', StandardScaler(), num),
        ('cols', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat),
    ],
    remainder='passthrough'
)

X = df.drop(columns=['RECURRENCE (LR, LRR)'])
Y = df[['RECURRENCE (LR, LRR)']]
for col in cat:
    X[col] = X[col].astype(str)

processed_data = preprocessor.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(processed_data, Y, train_size=0.8, random_state=42)
Y_train = Y_train.replace({'NO': 0, 'YES': 1}).astype(int)
Y_test = Y_test.replace({'NO': 0, 'YES': 1}).astype(int)

LR = LogisticRegression()
LR.fit(X_train, Y_train.values.ravel())

rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train, Y_train.values.ravel())

joblib.dump(LR, root / 'oscc_LR_model.joblib')
joblib.dump(rf_model, root / 'oscc_RF_model.joblib')
joblib.dump(preprocessor, root / 'preprocessor.joblib')
print('Saved oscc_LR_model.joblib, oscc_RF_model.joblib, preprocessor.joblib in root')
