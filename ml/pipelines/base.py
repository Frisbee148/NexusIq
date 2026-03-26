"""
Base pipeline class — enforces consistent ML workflow across all models.
Every pipeline (disruption, eta, anomaly) inherits from this and implements the abstract methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from utils.io import save_model, save_parquet, load_parquet, FEATURES_DIR, MODELS_DIR
from utils.metrics import save_eval_report
from utils.logger import logger
from utils.constants import RANDOM_SEED


class BasePipeline(ABC):
    """Abstract base class for all NexusIQ ML training pipelines."""

    def __init__(self, model_name: str, version: int = 1):
        self.model_name = model_name
        self.version = version
        self.model: Any = None
        self.metrics: dict = {}
        self.feature_names: list[str] = []

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load raw or synthetic data for this pipeline."""
        ...

    @abstractmethod
    def engineer_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        Transform raw data into feature matrix.
        Returns: (feature_dataframe, list_of_feature_column_names)
        """
        ...

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Any:
        """Train the model. Return the trained model object."""
        ...

    @abstractmethod
    def evaluate(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the model on test set. Return metrics dict."""
        ...

    @abstractmethod
    def get_target_column(self) -> str:
        """Return the name of the target column in the feature dataframe."""
        ...

    def split_data(self, df: pd.DataFrame, feature_cols: list[str],
                   test_size: float = 0.15, val_size: float = 0.15,
                   time_column: str | None = None) -> tuple:
        """
        Split data into train/val/test sets.
        If time_column is provided, uses time-based split (no future leakage).
        Otherwise, stratified random split.
        """
        target_col = self.get_target_column()
        X = df[feature_cols].values
        y = df[target_col].values

        if time_column and time_column in df.columns:
            # Time-based split
            df_sorted = df.sort_values(time_column)
            n = len(df_sorted)
            train_end = int(n * (1 - test_size - val_size))
            val_end = int(n * (1 - test_size))

            X_train = df_sorted[feature_cols].values[:train_end]
            y_train = df_sorted[target_col].values[:train_end]
            X_val = df_sorted[feature_cols].values[train_end:val_end]
            y_val = df_sorted[target_col].values[train_end:val_end]
            X_test = df_sorted[feature_cols].values[val_end:]
            y_test = df_sorted[target_col].values[val_end:]
        else:
            # Stratified random split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
            )
            relative_val = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=relative_val, random_state=RANDOM_SEED, stratify=y_temp
            )

        logger.info("data_split",
                     model=self.model_name,
                     train=len(X_train), val=len(X_val), test=len(X_test))
        return X_train, X_val, X_test, y_train, y_val, y_test

    def run(self) -> dict:
        """
        Execute the full pipeline: load → features → split → train → evaluate → save.
        Returns the evaluation metrics.
        """
        logger.info("pipeline_start", model=self.model_name, version=self.version)

        # 1. Load data
        df = self.load_data()
        logger.info("data_loaded", rows=len(df), columns=len(df.columns))

        # 2. Engineer features
        df_features, feature_cols = self.engineer_features(df)
        self.feature_names = feature_cols
        logger.info("features_engineered", feature_count=len(feature_cols))

        # Save feature matrix
        features_path = FEATURES_DIR / f"{self.model_name}_features.parquet"
        save_parquet(df_features, features_path)

        # 3. Split
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            df_features, feature_cols
        )

        # 4. Train
        self.model = self.train(X_train, y_train, X_val, y_val)
        logger.info("model_trained", model=self.model_name)

        # 5. Evaluate
        self.metrics = self.evaluate(self.model, X_test, y_test)
        logger.info("model_evaluated", model=self.model_name, metrics=self.metrics)

        # 6. Save
        save_model(
            model=self.model,
            model_name=self.model_name,
            version=self.version,
            metrics=self.metrics,
            hyperparams=getattr(self, "best_params", {}),
            dataset_path=features_path,
        )
        save_eval_report(self.metrics, self.model_name, self.version)

        logger.info("pipeline_complete", model=self.model_name, version=self.version)
        return self.metrics
