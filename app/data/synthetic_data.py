import numpy as np
import pandas as pd
import json
import os
from scipy import stats as sp_stats
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

CALIBRATION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', 'calibration_params.json'
)

# Ordered feature names matching the 25-feature spec
FEATURE_NAMES = [
    'age', 'systolicBP', 'diastolicBP', 'bloodGlucose', 'bodyTemp',
    'heartRate', 'bmi', 'hemoglobin', 'plateletCount', 'wbcCount',
    'gestationalAge', 'creatinine', 'alt', 'ast', 'proteinUrine',
    'hba1c', 'cholesterol', 'ldl', 'hdl', 'triglycerides',
    'ironLevel', 'vitaminD', 'thyroidTSH', 'previousPregnancies',
    'previousComplications'
]

class MaternalHealthDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def _sample_feature(params, n, rng):
    """Sample n values for a single feature from its calibrated distribution."""
    dist = params.get('dist', 'norm')
    mu = params['mu']
    std = params['std']

    if dist == 'truncnorm':
        lo, hi = params['bounds']
        a, b = (lo - mu) / std, (hi - mu) / std
        return sp_stats.truncnorm.rvs(a, b, loc=mu, scale=std, size=n, random_state=rng)
    else:
        return rng.normal(mu, std, size=n)


def _generate_calibrated(n_samples, random_state):
    """Generate data using calibration_params.json from the NCHS pipeline."""
    with open(CALIBRATION_PATH, 'r') as f:
        cal = json.load(f)

    rng = np.random.RandomState(random_state)
    df = pd.DataFrame()

    # 1. Sample each feature independently
    for feat in FEATURE_NAMES:
        if feat == 'previousComplications':
            # Binary feature — not in calibration, generate ~12% prevalence
            df[feat] = rng.binomial(1, 0.12, size=n_samples).astype(float)
        elif feat in cal and isinstance(cal[feat], dict):
            df[feat] = _sample_feature(cal[feat], n_samples, rng)
        else:
            df[feat] = rng.normal(0, 1, size=n_samples)

    # 2. Apply correlation shifts from calibration
    corr = cal.get('_correlations', {})
    bmi_z = (df['bmi'] - df['bmi'].mean()) / df['bmi'].std()

    if 'bmi_to_systolicBP' in corr:
        df['systolicBP'] += corr['bmi_to_systolicBP'] * bmi_z * cal['systolicBP']['std']
    if 'bmi_to_bloodGlucose' in corr:
        df['bloodGlucose'] += corr['bmi_to_bloodGlucose'] * bmi_z * cal['bloodGlucose']['std']
    if 'gestationalAge_to_bodyTemp' in corr:
        ga_z = (df['gestationalAge'] - df['gestationalAge'].mean()) / df['gestationalAge'].std()
        df['bodyTemp'] += corr['gestationalAge_to_bodyTemp'] * ga_z * cal['bodyTemp']['std']

    # 3. Derive high_risk label from clinical thresholds
    #    A patient is high-risk if ANY major risk factor is present
    risk_score = (
        (df['systolicBP'] > 140).astype(int) +
        (df['diastolicBP'] > 90).astype(int) +
        (df['bloodGlucose'] > 140).astype(int) +
        (df['proteinUrine'] > 300).astype(int) +
        (df['hba1c'] > 6.5).astype(int) +
        (df['hemoglobin'] < 10).astype(int) +
        (df['plateletCount'] < 100).astype(int) +
        (df['creatinine'] > 1.0).astype(int) +
        ((df['age'] > 35) & (df['previousComplications'] == 1)).astype(int)
    )
    df['high_risk'] = (risk_score >= 1).astype(int)

    return df


from app.data.natality_loader import NatalityMicrodataLoader

def _get_nchs_file():
    """Find unzipped NCHS txt file."""
    nchs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'nchs', 'natality')
    if not os.path.exists(nchs_dir):
        return None
    txt_files = [f for f in os.listdir(nchs_dir) if f.endswith('.txt')]
    return os.path.join(nchs_dir, txt_files[0]) if txt_files else None

def generate_synthetic_maternal_data(n_samples=1000, n_features=25, random_state=42):
    """
    Generates data using real NCHS human records as seeds.
    Base demographics (age, bmi, etc) are real; clinical labs are calibrated.
    """
    nchs_path = _get_nchs_file()
    
    if nchs_path:
        # 1. Load Real Human Records
        loader = NatalityMicrodataLoader(nchs_path, year=2022)
        # Load slightly more than requested to allow for filtering
        base_df, _ = loader.load(nrows=n_samples * 2) 
        base_df = base_df.sample(n=n_samples, random_state=random_state)
        
        # 2. Load Calibration for Supplementation
        with open(CALIBRATION_PATH, 'r') as f:
            cal = json.load(f)
        
        rng = np.random.RandomState(random_state)
        
        # 3. Fill missing clinical features (Labs, BP, Blood Glucose)
        for feat in base_df.columns:
            if base_df[feat].isnull().all():
                if feat in cal:
                    base_df[feat] = _sample_feature(cal[feat], n_samples, rng)
                else:
                    base_df[feat] = rng.normal(0, 1, size=n_samples)
        
        # 4. Fill local NaNs with column means
        base_df = base_df.fillna(base_df.mean())
        
        # 5. Labeling logic (High Risk based on real features + supplemented labs)
        risk_score = (
            (base_df['systolicBP'] > 140).astype(int) +
            (base_df['bloodGlucose'] > 140).astype(int) +
            (base_df['bmi'] > 35).astype(int) +
            (base_df['previousComplications'] == 1).astype(int)
        )
        base_df['high_risk'] = (risk_score >= 1).astype(int)
        
        return base_df[FEATURE_NAMES + ['high_risk']]
    
    # Fallback to pure calibrated generation if NCHS txt is missing
    if os.path.exists(CALIBRATION_PATH):
        return _generate_calibrated(n_samples, random_state)

    # Fallback: uncalibrated sklearn-based generation
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=2,
        weights=[0.85, 0.15],
        random_state=random_state
    )

    feature_names = [
        'age', 'bmi', 'systolic_bp', 'diastolic_bp', 'heart_rate',
        'blood_sugar', 'hemoglobin', 'platelet_count', 'wbc_count',
        'rbc_count', 'blood_urea', 'serum_creatinine', 'thyroid_tsh',
        'thyroid_t3', 'thyroid_t4', 'vitamin_d', 'calcium', 'iron',
        'uric_acid', 'cholesterol', 'triglycerides', 'hdl', 'ldl',
        'previous_c_section', 'parity'
    ]

    df = pd.DataFrame(X, columns=feature_names[:n_features])
    df['age'] = np.clip(df['age'] * 10 + 25, 18, 45)
    df['bmi'] = np.clip(df['bmi'] * 5 + 25, 18, 40)
    df['systolic_bp'] = np.clip(df['systolic_bp'] * 20 + 120, 90, 160)
    df['diastolic_bp'] = np.clip(df['diastolic_bp'] * 15 + 80, 60, 100)
    df['heart_rate'] = np.clip(df['heart_rate'] * 15 + 85, 60, 120)
    df['high_risk'] = y

    return df

def split_data_for_federated_learning(df, n_hospitals=3, test_size=0.2, random_state=42):
    """
    Split data into multiple hospital datasets for federated learning
    """
    # First split into train and test
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['high_risk']
    )
    
    # Split training data among hospitals
    hospital_dfs = []
    hospital_size = len(train_df) // n_hospitals
    
    for i in range(n_hospitals):
        start_idx = i * hospital_size
        end_idx = (i + 1) * hospital_size if i < n_hospitals - 1 else len(train_df)
        hospital_df = train_df.iloc[start_idx:end_idx].copy()
        hospital_dfs.append(hospital_df)
    
    return hospital_dfs, test_df

def prepare_dataloaders(hospital_dfs, test_df, batch_size=32):
    """
    Prepare PyTorch dataloaders for each hospital and test set.
    Features are standardized (zero mean, unit variance) using training statistics.
    """
    # Compute standardization stats from all training data
    all_train = pd.concat(hospital_dfs)
    train_features_all = all_train.drop('high_risk', axis=1).values.astype(np.float32)
    feat_mean = train_features_all.mean(axis=0)
    feat_std = train_features_all.std(axis=0)
    feat_std[feat_std == 0] = 1.0  # avoid division by zero

    hospital_dataloaders = []

    for hospital_df in hospital_dfs:
        features = hospital_df.drop('high_risk', axis=1).values.astype(np.float32)
        features = (features - feat_mean) / feat_std
        labels = hospital_df['high_risk'].values.astype(np.float32)

        dataset = MaternalHealthDataset(
            torch.tensor(features),
            torch.tensor(labels).unsqueeze(1)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        hospital_dataloaders.append(dataloader)

    # Prepare test dataloader (standardized with training stats)
    test_features = test_df.drop('high_risk', axis=1).values.astype(np.float32)
    test_features = (test_features - feat_mean) / feat_std
    test_labels = test_df['high_risk'].values.astype(np.float32)

    test_dataset = MaternalHealthDataset(
        torch.tensor(test_features),
        torch.tensor(test_labels).unsqueeze(1)
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Compute pos_weight for class imbalance
    all_labels = all_train['high_risk']
    n_pos = all_labels.sum()
    n_neg = len(all_labels) - n_pos
    pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

    return hospital_dataloaders, test_dataloader, pos_weight
