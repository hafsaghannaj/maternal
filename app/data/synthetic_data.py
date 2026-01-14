import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class MaternalHealthDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def generate_synthetic_maternal_data(n_samples=1000, n_features=25, random_state=42):
    """
    Generate synthetic maternal health data with realistic features
    """
    # Base features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=2,
        weights=[0.85, 0.15],  # 15% high risk
        random_state=random_state
    )
    
    # Create feature names that mimic real maternal health indicators
    feature_names = [
        'age', 'bmi', 'systolic_bp', 'diastolic_bp', 'heart_rate',
        'blood_sugar', 'hemoglobin', 'platelet_count', 'wbc_count',
        'rbc_count', 'blood_urea', 'serum_creatinine', 'thyroid_tsh',
        'thyroid_t3', 'thyroid_t4', 'vitamin_d', 'calcium', 'iron',
        'uric_acid', 'cholesterol', 'triglycerides', 'hdl', 'ldl',
        'previous_c_section', 'parity'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names[:n_features])
    
    # Make data more realistic by adjusting ranges
    df['age'] = np.clip(df['age'] * 10 + 25, 18, 45)  # Age between 18-45
    df['bmi'] = np.clip(df['bmi'] * 5 + 25, 18, 40)   # BMI between 18-40
    df['systolic_bp'] = np.clip(df['systolic_bp'] * 20 + 120, 90, 160)
    df['diastolic_bp'] = np.clip(df['diastolic_bp'] * 15 + 80, 60, 100)
    df['heart_rate'] = np.clip(df['heart_rate'] * 15 + 85, 60, 120)
    
    # Add target
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
    Prepare PyTorch dataloaders for each hospital and test set
    """
    hospital_dataloaders = []
    
    for hospital_df in hospital_dfs:
        # Separate features and labels
        features = hospital_df.drop('high_risk', axis=1).values.astype(np.float32)
        labels = hospital_df['high_risk'].values.astype(np.float32)
        
        # Create dataset and dataloader
        dataset = MaternalHealthDataset(
            torch.tensor(features),
            torch.tensor(labels).unsqueeze(1)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        hospital_dataloaders.append(dataloader)
    
    # Prepare test dataloader
    test_features = test_df.drop('high_risk', axis=1).values.astype(np.float32)
    test_labels = test_df['high_risk'].values.astype(np.float32)
    
    test_dataset = MaternalHealthDataset(
        torch.tensor(test_features),
        torch.tensor(test_labels).unsqueeze(1)
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return hospital_dataloaders, test_dataloader
