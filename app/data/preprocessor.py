import numpy as np
from sklearn.preprocessing import StandardScaler


def split_features_labels(df, label_col="high_risk"):
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe.")
    features = df.drop(columns=[label_col])
    labels = df[label_col]
    return features, labels


def fit_scaler(train_df, feature_cols=None):
    if feature_cols is None:
        feature_cols = train_df.columns.tolist()
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler, feature_cols


def transform_features(df, scaler, feature_cols):
    if scaler is None:
        raise ValueError("Scaler is required to transform features.")
    features = scaler.transform(df[feature_cols])
    return features.astype(np.float32)


def prepare_features(
    df,
    label_col="high_risk",
    scaler=None,
    fit_scaler=False,
    feature_cols=None,
):
    features_df, labels = split_features_labels(df, label_col=label_col)
    if feature_cols is None:
        feature_cols = features_df.columns.tolist()

    if fit_scaler:
        scaler, feature_cols = fit_scaler(features_df, feature_cols=feature_cols)

    if scaler is None:
        features = features_df[feature_cols].values.astype(np.float32)
    else:
        features = transform_features(features_df, scaler, feature_cols)

    return features, labels.values.astype(np.float32), scaler, feature_cols
