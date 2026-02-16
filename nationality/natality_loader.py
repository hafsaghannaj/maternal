import pandas as pd
import numpy as np
import os
import requests
from typing import Tuple, Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NatalityMicrodataLoader:
    """
    Parser for NCHS Natality fixed-width microdata files.
    Supports layout for 2022/2023 (2003 revised certificate).
    """
    
    # 2023 Layout Specs (Based on User Guide)
    # Note: These are 1-indexed positions for documentation, 
    # but pandas read_fwf uses 0-indexed half-open intervals [start, end)
    # So Position 9-12 becomes (8, 12)
    COLUMN_SPECS_2023 = {
        'DOB_YY': (8, 12),
        'MAGER': (74, 76),
        'MRACEHISP': (116, 117), # Position 117
        'MEDUC': (123, 124),     # Position 124
        'PRECARE': (223, 225),   # Position 224-225
        'CIG_0': (252, 254),     # Position 253-254
        'BMI': (282, 286),       # Position 283-286
        'WTGAIN': (303, 305),    # Position 304-305
        'RF_PPDIAB': (312, 313), # Position 313
        'RF_GDIAB': (313, 314),  # Position 314
        'RF_PPHYPE': (314, 315), # Position 315
        'RF_GHYPE': (315, 316),  # Position 316
        'RF_ECLAMP': (316, 317), # Position 317
        'RF_CESARN': (330, 332), # Position 331-332
        'DMETH_REC': (407, 408), # Position 408
        'MM_APTS': (418, 419),   # Position 419 (AICU)
        'PAY': (434, 435),       # Position 435
        'OE_GEST': (498, 500),   # Position 499-500
        'DBWT': (503, 507),      # Position 504-507
        'ILIVE': (435, 436),     # Position 436
    }

    FEATURES_25_SPEC = [
        'age', 'systolicBP', 'diastolicBP', 'bloodGlucose', 'bodyTemp', 
        'heartRate', 'bmi', 'hemoglobin', 'plateletCount', 'wbcCount', 
        'gestationalAge', 'creatinine', 'alt', 'ast', 'proteinUrine', 
        'hba1c', 'cholesterol', 'ldl', 'hdl', 'triglycerides', 
        'ironLevel', 'vitaminD', 'thyroidTSH', 'previousPregnancies', 
        'previousComplications'
    ]

    def __init__(self, file_path: str, year: int = 2023):
        self.file_path = file_path
        self.year = year
        self.colspecs = self.COLUMN_SPECS_2023 if year >= 2022 else self._get_legacy_specs(year)

    def _get_legacy_specs(self, year: int) -> Dict[str, Tuple[int, int]]:
        # Placeholder for older years if needed
        return self.COLUMN_SPECS_2023 

    def load(self, nrows: int = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Loads and parses the fixed-width file.
        Returns (features_df, metadata).
        """
        logger.info(f"Parsing NCHS Natality file: {self.file_path} (Year: {self.year})")
        
        # Prepare colspecs and names for pandas
        names = list(self.colspecs.keys())
        specs = list(self.colspecs.values())
        
        df = pd.read_fwf(
            self.file_path,
            colspecs=specs,
            names=names,
            nrows=nrows,
            dtype=str
        )
        
        # Data Cleaning & Mapping
        processed_df = self._process_data(df)
        
        # Map to 25-feature vector
        features_df = self._map_to_25_features(processed_df)
        
        # Generate DataQualityReport
        report = self._generate_report(features_df)
        
        metadata = {
            "year": self.year,
            "total_records": len(df),
            "quality_report": report,
            "supplementation_needed": [f for f in self.FEATURES_25_SPEC if features_df[f].isnull().all()]
        }
        
        return features_df, metadata

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize and clean raw fields."""
        # Age
        df['age'] = pd.to_numeric(df['MAGER'], errors='coerce')
        
        # BMI (Format is ##.#)
        df['bmi'] = pd.to_numeric(df['BMI'], errors='coerce') / 10.0
        df.loc[df['bmi'] > 99, 'bmi'] = np.nan # 999 is usually unknown
        
        # Gestational Age
        df['gestationalAge'] = pd.to_numeric(df['OE_GEST'], errors='coerce')
        df.loc[df['gestationalAge'] > 98, 'gestationalAge'] = np.nan
        
        # Birth Order (Previous Pregnancies estimate)
        df['previousPregnancies'] = pd.to_numeric(df['ILIVE'], errors='coerce') - 1
        df.loc[df['previousPregnancies'] < 0, 'previousPregnancies'] = 0
        
        # Risk Flags (Y=1, N=2, U=9)
        risk_cols = ['RF_PPDIAB', 'RF_GDIAB', 'RF_PPHYPE', 'RF_GHYPE', 'RF_ECLAMP']
        for col in risk_cols:
            df[col] = df[col].map({'1': 1, '2': 0}).fillna(0) # Simplify to binary
            
        # Composite feature for previousComplications
        df['previousComplications'] = (
            df['RF_PPDIAB'] + df['RF_PPHYPE'] + 
            (pd.to_numeric(df['RF_CESARN'], errors='coerce').fillna(0) > 0).astype(int)
        ).clip(0, 1)
        
        return df

    def _map_to_25_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map cleaned data to the standardized 25-feature vector."""
        out_df = pd.DataFrame(index=df.index)
        
        # Direct mappings
        out_df['age'] = df['age']
        out_df['bmi'] = df['bmi']
        out_df['gestationalAge'] = df['gestationalAge']
        out_df['previousPregnancies'] = df['previousPregnancies']
        out_df['previousComplications'] = df['previousComplications']
        
        # Initialize missing features as NaN
        missing_features = [f for f in self.FEATURES_25_SPEC if f not in out_df.columns]
        for f in missing_features:
            out_df[f] = np.nan
            
        return out_df[self.FEATURES_25_SPEC]

    def _generate_report(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate missing rates per feature."""
        missing_rates = (df.isnull().sum() / len(df)) * 100
        return missing_rates.to_dict()

def download_natality_file(year: int, target_dir: str):
    """
    Downloads the US natality file from CDC FTP.
    Example: ftp://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/DVS/natality/Nat2022us.zip
    """
    import urllib.request
    
    os.makedirs(target_dir, exist_ok=True)
    filename = f"Nat{year}us.zip"
    url = f"ftp://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/DVS/natality/{filename}"
    target_path = os.path.join(target_dir, filename)
    
    logger.info(f"Downloading {url} to {target_path}...")
    try:
        urllib.request.urlretrieve(url, target_path)
        logger.info(f"Successfully downloaded {filename}")
    except Exception as e:
        logger.error(f"Failed to download {filename}: {e}")
        raise
