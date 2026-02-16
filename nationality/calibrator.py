import pandas as pd
import numpy as np
import scipy.stats as stats
import json
import os
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class CalibrateSyntheticData:
    """
    Calibrates synthetic data generators using real-world distributions 
    from NCHS, CDC WONDER, and clinical literature.
    """
    
    # Clinical Reference Ranges (Mean, STD approximation)
    REFERENCE_RANGES = {
        'systolicBP': {'mean': 115, 'std': 12, 'bounds': (90, 160)},
        'diastolicBP': {'mean': 75, 'std': 10, 'bounds': (60, 100)},
        'bloodGlucose': {'mean': 85, 'std': 15, 'bounds': (40, 200)},
        'bodyTemp': {'mean': 98.6, 'std': 0.4, 'bounds': (97.0, 101.0)},
        'heartRate': {'mean': 80, 'std': 12, 'bounds': (60, 120)},
        'hemoglobin': {'mean': 12.5, 'std': 1.2, 'bounds': (8.0, 16.0)},
        'plateletCount': {'mean': 250, 'std': 60, 'bounds': (50, 600)},
        'wbcCount': {'mean': 11, 'std': 3, 'bounds': (4, 30)},
        'creatinine': {'mean': 0.6, 'std': 0.15, 'bounds': (0.3, 1.5)},
        'alt': {'mean': 15, 'std': 8, 'bounds': (3, 100)},
        'ast': {'mean': 15, 'std': 8, 'bounds': (3, 100)},
        'proteinUrine': {'mean': 50, 'std': 40, 'bounds': (0, 5000)},
        'hba1c': {'mean': 5.0, 'std': 0.4, 'bounds': (4.0, 10.0)},
        'cholesterol': {'mean': 220, 'std': 40, 'bounds': (120, 400)},
        'ldl': {'mean': 130, 'std': 30, 'bounds': (60, 250)},
        'hdl': {'mean': 65, 'std': 15, 'bounds': (30, 100)},
        'triglycerides': {'mean': 180, 'std': 60, 'bounds': (50, 500)},
        'ironLevel': {'mean': 100, 'std': 30, 'bounds': (30, 250)},
        'vitaminD': {'mean': 35, 'std': 10, 'bounds': (10, 100)},
        'thyroidTSH': {'mean': 1.5, 'std': 0.8, 'bounds': (0.1, 10.0)}
    }

    def __init__(self, output_path: str = "./config/calibration_params.json"):
        self.output_path = output_path
        self.params = {}
        self.report = {}

    def run_calibration(self, 
                       natality_df: Optional[pd.DataFrame] = None, 
                       cdc_data: Optional[Dict[str, Any]] = None,
                       ahr_data: Optional[Dict[str, Any]] = None):
        """
        Fits distributions and saves parameters.
        """
        logger.info("Starting data calibration...")
        
        # 1. Calibrate from Real Microdata (NCHS)
        if natality_df is not None:
            self._calibrate_from_natality(natality_df)
            
        # 2. Integrate CDC WONDER & AHR Prevalence Rates
        if cdc_data:
            self._integrate_cdc_prevalence(cdc_data)
        if ahr_data:
            self._integrate_ahr_prevalence(ahr_data)

        # 3. Fill in Clinical Reference Ranges for Lab Values
        self._set_clinical_defaults()

        # 4. Define Correlation Structure
        self._define_correlations()

        # 5. Save and Export
        self._save_params()
        self._generate_report(natality_df)
        
        return self.report

    def _calibrate_from_natality(self, df: pd.DataFrame):
        features = ['age', 'bmi', 'gestationalAge', 'previousPregnancies']
        for feat in features:
            if feat in df.columns:
                valid_data = df[feat].dropna()
                if not valid_data.empty:
                    mu, std = stats.norm.fit(valid_data)
                    self.params[feat] = {
                        'dist': 'norm',
                        'mu': float(mu),
                        'std': float(std),
                        'source': 'NCHS Natality'
                    }

    def _integrate_cdc_prevalence(self, cdc_data: Dict[str, Any]):
        # Example: Adjusting risk factors like diabetes/hypertension
        for factor, rate in cdc_data.items():
            self.params[factor] = {
                'dist': 'bernoulli',
                'p': float(rate),
                'source': 'CDC WONDER'
            }

    def _integrate_ahr_prevalence(self, ahr_data: Dict[str, Any]):
        # Example: Racial disparities or state-level offsets
        pass

    def _set_clinical_defaults(self):
        """Apply truncated normal parameters for features without real data."""
        for feat, ref in self.REFERENCE_RANGES.items():
            if feat not in self.params:
                self.params[feat] = {
                    'dist': 'truncnorm',
                    'mu': ref['mean'],
                    'std': ref['std'],
                    'bounds': ref['bounds'],
                    'source': 'Clinical Reference'
                }

    def _define_correlations(self):
        """
        Defines how features interact (e.g., high BMI -> higher BP).
        Stored as a correlation matrix or specific shift weights.
        """
        self.params['_correlations'] = {
            'bmi_to_systolicBP': 0.4,
            'bmi_to_bloodGlucose': 0.35,
            'preeclampsia_to_systolicBP_shift': 30,
            'preeclampsia_to_proteinUrine_shift': 500,
            'gestationalAge_to_bodyTemp': 0.1
        }

    def _save_params(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(self.params, f, indent=4)
        logger.info(f"Calibration parameters saved to {self.output_path}")

    def _generate_report(self, natality_df: Optional[pd.DataFrame]):
        self.report = {
            "summary": "Calibration complete",
            "features_calibrated": list(self.params.keys()),
            "sources": list(set(v.get('source') for v in self.params.values() if isinstance(v, dict)))
        }
        # Add comparisons if natality_df is available
        if natality_df is not None:
            self.report["natality_samples"] = len(natality_df)
