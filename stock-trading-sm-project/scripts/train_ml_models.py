import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
import logging
from pathlib import Path
import joblib

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception as _e:  # pragma: no cover
    lgb = None
    HAS_LGBM = False
    logging.getLogger(__name__).warning(f"LightGBM unavailable: {_e}")

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, mean_absolute_error, mean_squared_error,
    roc_auc_score, log_loss
)
from sklearn.inspection import permutation_importance

from sklearn.preprocessing import LabelEncoder
try:
    import shap  # noqa: F401
except Exception:
    shap = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLModelTrainer:
    """Train and evaluate ML models for stock trading"""

    def __init__(self, seed: int = 42):
        """
        Initialize model trainer

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.models = {}
        self.best_params = {}
        self.metrics = {}
        self.label_encoder = None

        np.random.seed(seed)

        # Define parameter grids for different models
        self.param_grids = {
            'decision_tree': {
                'max_depth': [5, 10, 20],
                'min_samples_split': [2, 10],
                'min_samples_leaf': [1, 4],
            },
            'random_forest': {
                'n_estimators': [100],
                'max_depth': [5, 10],
                'min_samples_leaf': [1, 2],
            },
            'xgboost': {
                'n_estimators': [100],
                'max_depth': [3, 6],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.9],
            },
            'lightgbm': {
                'num_leaves': [31, 50],
                'max_depth': [-1, 10],
                'learning_rate': [0.05, 0.1],
                'n_estimators': [100],
            }
        }

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'target',
        feature_cols: List[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
               Dict[str, LabelEncoder]]:
        """
        Prepare features and targets for model training

        Args:
            df: DataFrame
            target_col: Column name for target
            feature_cols: List of feature columns

        Returns:
            X_train, X_test, y_train, y_test encoders
        """
        logger.info("Preparing data for model training")

        if feature_cols is None:
            feature_cols = df.columns.tolist()

        # Filter to only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude target and known non-feature / leakage columns. Benchmark
        # columns are kept only for the buy & hold comparison, not as features.
        exclude_cols = ['symbol', target_col, 'continuous_target', 'future_returns', 'date']
        feature_cols = [
            col for col in numeric_cols
            if col not in exclude_cols and not col.startswith('BENCHMARK_')
        ]

        logger.info(f"Using {len(feature_cols)} features for modeling")

        # Encode symbol column if exists
        encoders = {}
        if 'symbol' in df.columns:
            le = LabelEncoder()
            df['symbol_encoded'] = le.fit_transform(df['symbol'])
            encoders['symbol'] = le
            feature_cols.append('symbol_encoded')

        # Split data into training and testing. Use a date boundary so that
        # no single date is shared between train and test.
        X = df[feature_cols].copy().ffill().bfill()
        y = df[target_col].copy()

        if 'date' in df.columns:
            dates = np.sort(df['date'].unique())
            split_date = dates[int(len(dates) * 0.8)]
            train_mask = (df['date'] < split_date).to_numpy()
        else:
            train_size = int(len(df) * 0.8)
            train_mask = np.zeros(len(df), dtype=bool)
            train_mask[:train_size] = True

        X_train = X[train_mask]
        X_test = X[~train_mask]
        y_train = y[train_mask]
        y_test = y[~train_mask]

        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Testing data shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test, encoders

    def train_decision_tree_classification(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv_splits: int = 5
    ) -> DecisionTreeClassifier:
        """
        Train Decision Tree classifier with grid search

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Testing features
            y_test: Testing targets
            cv_splits: Number of cross-validation splits

        Returns:
            Trained Decision Tree classifier
        """
        logger.info("Training Decision Tree Classifier")

        dt = DecisionTreeClassifier(random_state=self.seed)

        try:
            # Grid search
            grid_search = GridSearchCV(
                estimator=dt,
                param_grid=self.param_grids['decision_tree'],
                scoring='f1_weighted',
                cv=TimeSeriesSplit(n_splits=cv_splits),
                n_jobs=-1,
                verbose=0
            )

            logger.info("Performing grid search with TimeSeriesSplit...")

            grid_search.fit(X_train, y_train)

            best_dt = grid_search.best_estimator_
            self.best_params['decision_tree'] = grid_search.best_params_

            logger.info(f"Best Decision Tree params: {grid_search.best_params_}")

            # Evaluate
            y_pred = best_dt.predict(X_test)
            y_prob = self._up_proba(best_dt, X_test)

            self.models['decision_tree'] = best_dt
            self.metrics['decision_tree'] = self._evaluate_model(
                y_test, y_pred, y_prob
            )

        except Exception as e:
            logger.error(f"Error training Decision Tree: {str(e)}")
            self.models['decision_tree'] = dt

        return self.models.get('decision_tree', dt)

    def train_random_forest_classification(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv_splits: int = 5
    ) -> RandomForestClassifier:
        """
        Train Random Forest classifier with grid search

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Testing features
            y_test: Testing targets
            cv_splits: Number of cross-validation splits

        Returns:
            Trained Random Forest classifier
        """
        logger.info("Training Random Forest Classifier")

        rf = RandomForestClassifier(random_state=self.seed, n_jobs=-1)

        try:
            grid_search = GridSearchCV(
                estimator=rf,
                param_grid=self.param_grids['random_forest'],
                scoring='f1_weighted',
                cv=TimeSeriesSplit(n_splits=cv_splits),
                n_jobs=-1,
                verbose=0
            )

            logger.info("Performing grid search with TimeSeriesSplit...")

            grid_search.fit(X_train, y_train)

            best_rf = grid_search.best_estimator_
            self.best_params['random_forest'] = grid_search.best_params_

            logger.info(f"Best Random Forest params: {grid_search.best_params_}")

            y_pred = best_rf.predict(X_test)
            y_prob = self._up_proba(best_rf, X_test)

            self.models['random_forest'] = best_rf
            self.metrics['random_forest'] = self._evaluate_model(
                y_test, y_pred, y_prob
            )

        except Exception as e:
            logger.error(f"Error training Random Forest: {str(e)}")
            self.models['random_forest'] = rf

        return self.models.get('random_forest', rf)

    def train_xgboost_classification(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv_splits: int = 5
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost classifier with grid search

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Testing features
            y_test: Testing targets
            cv_splits: Number of cross-validation splits

        Returns:
            Trained XGBoost classifier
        """
        logger.info("Training XGBoost Classifier")

        # XGBoost wants contiguous integer labels, so encode internally and
        # map predictions back to {-1, 0, 1} for evaluation.
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=self.seed,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        try:
            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=self.param_grids['xgboost'],
                scoring='f1_weighted',
                cv=TimeSeriesSplit(n_splits=cv_splits),
                n_jobs=-1,
                verbose=0
            )

            logger.info("Performing grid search with TimeSeriesSplit...")

            grid_search.fit(X_train, y_train_enc)

            best_xgb = grid_search.best_estimator_
            self.best_params['xgboost'] = grid_search.best_params_

            logger.info(f"Best XGBoost params: {grid_search.best_params_}")

            y_pred_enc = best_xgb.predict(X_test)
            y_pred = le.inverse_transform(y_pred_enc)

            proba = best_xgb.predict_proba(X_test)
            up_idx = list(le.classes_).index(1)
            y_prob = proba[:, up_idx]

            self.label_encoder = le
            self.models['xgboost'] = best_xgb
            self.metrics['xgboost'] = self._evaluate_model(
                y_test, pd.Series(y_pred), y_prob
            )

        except Exception as e:
            logger.error(f"Error training XGBoost: {str(e)}")
            self.models['xgboost'] = xgb_model

        return self.models.get('xgboost', xgb_model)

    def train_lightgbm_classification(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv_splits: int = 5
    ) -> Any:
        """
        Train LightGBM classifier with grid search

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Testing features
            y_test: Testing targets
            cv_splits: Number of cross-validation splits

        Returns:
            Trained LightGBM classifier
        """
        logger.info("Training LightGBM Classifier")

        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1
        )

        try:
            grid_search = GridSearchCV(
                estimator=lgb_model,
                param_grid=self.param_grids['lightgbm'],
                scoring='f1_weighted',
                cv=TimeSeriesSplit(n_splits=cv_splits),
                n_jobs=-1,
                verbose=-1
            )

            logger.info("Performing grid search with TimeSeriesSplit...")

            grid_search.fit(X_train, y_train)

            best_lgb = grid_search.best_estimator_
            self.best_params['lightgbm'] = grid_search.best_params_

            logger.info(f"Best LightGBM params: {grid_search.best_params_}")

            y_pred = best_lgb.predict(X_test)
            y_prob = self._up_proba(best_lgb, X_test)

            self.models['lightgbm'] = best_lgb
            self.metrics['lightgbm'] = self._evaluate_model(
                y_test, y_pred, y_prob
            )

        except Exception as e:
            logger.error(f"Error training LightGBM: {str(e)}")
            self.models['lightgbm'] = lgb_model

        return self.models.get('lightgbm', lgb_model)

    def train_lightning_boosting(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv_splits: int = 5
    ) -> xgb.XGBClassifier:
        """
        Train LightningBoosting (XGBoost with early stopping for speed)

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Testing features
            y_test: Testing targets
            cv_splits: Number of cross-validation splits

        Returns:
            Trained XGBoost model with early stopping
        """
        logger.info("Training LightningBoosting with early stopping")

        # Prepare validation data
        val_size = int(len(X_train) * 0.2)
        X_train_sub, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
        y_train_sub, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]

        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=self.seed,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss',
            early_stopping_rounds=50,
            verbosity=0
        )

        try:
            xgb_model.fit(
                X_train_sub, y_train_sub,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # Final test evaluation
            y_pred = xgb_model.predict(X_test)
            y_prob = self._up_proba(xgb_model, X_test)

            self.models['lightning_boosting'] = xgb_model
            self.metrics['lightning_boosting'] = self._evaluate_model(
                y_test, y_pred, y_prob
            )

            logger.info("LightningBoosting trained successfully")

        except Exception as e:
            logger.error(f"Error training LightningBoosting: {str(e)}")
            self.models['lightning_boosting'] = xgb_model

        return xgb_model

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        models_to_train: List[str] = None,
        cv_splits: int = 5
    ):
        """
        Train all configured models

        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training targets
            y_test: Testing targets
            models_to_train: List of models to train
            cv_splits: Number of cross-validation splits

        Returns:
            Dictionary of trained models
        """
        if models_to_train is None:
            models_to_train = list(self.param_grids.keys())

        logger.info(f"Training {len(models_to_train)} models: {models_to_train}")

        for model in models_to_train:
            if model == 'decision_tree':
                self.train_decision_tree_classification(X_train, y_train, X_test, y_test, cv_splits)
            elif model == 'random_forest':
                self.train_random_forest_classification(X_train, y_train, X_test, y_test, cv_splits)
            elif model == 'xgboost':
                self.train_xgboost_classification(X_train, y_train, X_test, y_test, cv_splits)
            elif model == 'lightgbm':
                if not HAS_LGBM:
                    logger.warning("Skipping LightGBM (not available)")
                    continue
                self.train_lightgbm_classification(X_train, y_train, X_test, y_test, cv_splits)
            elif model == 'lightning_boosting':
                self.train_lightning_boosting(X_train, y_train, X_test, y_test, cv_splits)

        logger.info("All models training completed")

        return self.models

    def _up_proba(self, model, X: pd.DataFrame) -> np.ndarray:
        """Return probability of the 'up' class (label 1) from a fitted classifier."""
        proba = model.predict_proba(X)
        classes = list(getattr(model, 'classes_', []))
        if 1 in classes:
            return proba[:, classes.index(1)]
        # fallback: last column
        return proba[:, -1]

    def _evaluate_model(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_prob: pd.Series = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance (multiclass-safe)

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probability of the 'up' (class 1) event, optional

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }

        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                binary_up = (np.asarray(y_true) == 1).astype(int)
                if len(np.unique(binary_up)) > 1:
                    metrics['roc_auc'] = roc_auc_score(binary_up, y_prob)
                else:
                    metrics['roc_auc'] = 0.0
            except Exception:
                metrics['roc_auc'] = 0.0
        else:
            metrics['roc_auc'] = 0.0

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        for cls, name in [('1', 'up'), ('-1', 'down'), ('0', 'hold')]:
            key = cls
            if key not in report:
                # keys may be ints
                try:
                    key = int(cls)
                except ValueError:
                    key = cls
            entry = report.get(key, {})
            metrics[f'precision_{name}'] = entry.get('precision', 0.0)
            metrics[f'recall_{name}'] = entry.get('recall', 0.0)
            metrics[f'f1_{name}'] = entry.get('f1-score', 0.0)

        return metrics


    def predict_full(self, model_name: str, X: pd.DataFrame):
        """Return (pred_class, prob_up, prob_down) in original label space."""
        model = self.models[model_name]
        proba = model.predict_proba(X)
        classes = list(model.classes_)

        if model_name == 'xgboost' and self.label_encoder is not None:
            le = self.label_encoder
            pred = le.inverse_transform(model.predict(X))
            up_idx = list(le.classes_).index(1)
            down_idx = list(le.classes_).index(-1)
        else:
            pred = model.predict(X)
            up_idx = classes.index(1) if 1 in classes else -1
            down_idx = classes.index(-1) if -1 in classes else None

        prob_up = proba[:, up_idx]
        prob_down = proba[:, down_idx] if down_idx is not None else np.zeros(len(X))
        return pred, prob_up, prob_down

    def select_best_model(self, return_name: bool = True):
        """
        Select the best performing model based on F1 score

        Args:
            return_name: Return just the model name if True, else return entire model

        Returns:
            Best model or model name
        """
        best_model_name = max(
            self.metrics.keys(),
            key=lambda k: self.metrics[k].get('f1_score', 0)
        )

        logger.info(f"Selected best model: {best_model_name}")

        if return_name:
            return best_model_name
        else:
            return self.models[best_model_name]

    def get_model_performance_table(self) -> pd.DataFrame:
        """
        Create performance comparison table for all models

        Returns:
            DataFrame with model performance metrics
        """
        performance_data = {}

        for model_name, metrics in self.metrics.items():
            performance_data[model_name] = {
                'accuracy': f"{metrics['accuracy']:.4f}",
                'precision': f"{metrics['precision']:.4f}",
                'recall': f"{metrics['recall']:.4f}",
                'f1_score': f"{metrics['f1_score']:.4f}",
                'roc_auc': f"{metrics['roc_auc']:.4f}",
                'precision_up': f"{metrics['precision_up']:.4f}",
                'recall_up': f"{metrics['recall_up']:.4f}",
                'precision_down': f"{metrics['precision_down']:.4f}",
                'recall_down': f"{metrics['recall_down']:.4f}"
            }

        df = pd.DataFrame(performance_data).T
        df.index.name = 'Model'

        return df

    def save_models(self, output_dir: str = "models"):
        """
        Save trained models and metrics to disk

        Args:
            output_dir: Directory to save models
        """
        logger.info(f"Saving models to {output_dir}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save models
        for model_name, model in self.models.items():
            model_path = Path(output_dir) / f"{model_name}.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} to {model_path}")

        # Save best parameters
        params_path = Path(output_dir) / "best_params.json"
        import json
        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=4)
        logger.info(f"Saved best parameters to {params_path}")

        # Save metrics
        metrics_path = Path(output_dir) / "model_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                model_name: {k: float(v) if isinstance(v, str) else v
                          for k, v in metrics.items()}
                for model_name, metrics in self.metrics.items()
            }, f, indent=4)
        logger.info(f"Saved model metrics to {metrics_path}")

        # Save performance table
        perf_table = self.get_model_performance_table()
        perf_path = Path(output_dir) / "performance_table.csv"
        perf_table.to_csv(perf_path)
        logger.info(f"Saved performance table to {perf_path}")

    def plot_feature_importance(
        self,
        model: xgb.XGBClassifier = None,
        feature_names: List[str] = None,
        max_features: int = 15,
        output_path: str = None
    ):
        """
        Plot feature importance for XGBoost/RandomForest models

        Args:
            model: Model to analyze (uses best if None)
            feature_names: List of feature names
            max_features: Maximum number of features to show
            output_path: Path to save plot
        """
        if model is None:
            model = self.models['xgboost']

        if feature_names is None:
            feature_names = ['Feature'][:max_features]

        try:
            # Get feature importance
            try:
                importance = model.feature_importances_ if hasattr(model, 'feature_importances_') else None

                if importance is not None:
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importance
                    }).sort_values('Importance', ascending=False).head(max_features)

                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(12, 8))
                    plt.barh(
                        importance_df['Feature'],
                        importance_df['Importance'],
                        color='steelblue'
                    )
                    plt.xlabel('Feature Importance')
                    plt.title('Feature Importance Ranking')
                    plt.tight_layout()

                    if output_path:
                        plt.savefig(output_path, dpi=300, bbox_inches='tight')
                        logger.info(f"Saved feature importance plot to {output_path}")

                    plt.close()

            except Exception as e:
                logger.error(f"Error plotting feature importance: {str(e)}")

        except Exception as e:
            logger.error(f"Error in feature importance analysis: {str(e)}")

    def cross_validate_model(
        self,
        model_name: str,
        cv_splits: int = 10,
        n_repeats: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Perform time-series cross-validation

        Args:
            model_name: Name of model to validate
            cv_splits: Number of folds
            n_repeats: Number of times to repeat cross-validation

        Returns:
            Dictionary with CV results
        """
        logger.info(f"Performing cross-validation for {model_name}")

        model = self.models.get(model_name)

        if model is None:
            logger.warning(f"Model {model_name} not trained")
            return {}

        cv_results = []

        tscv = TimeSeriesSplit(n_splits=cv_splits)

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_train)):
            X_train_fold, X_test_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_train_fold, y_test_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]

            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_test_fold)
            y_prob = self._up_proba(model, X_test_fold)

            fold_metrics = self._evaluate_model(y_test_fold, y_pred, y_prob)
            fold_metrics['fold'] = fold + 1

            cv_results.append(fold_metrics)

        cv_df = pd.DataFrame(cv_results)

        logger.info(f"Cross-validation completed for {model_name}")
        logger.info(f"CV Accuracy: {cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f}")

        return {'cv_results': cv_df, 'cv_mean': cv_df.mean(), 'cv_std': cv_df.std()}

    def save_feature_importance(self, X_test, y_test, output_dir):
        """Save built-in and permutation importance for the selected model."""
        best_name = self.select_best_model(return_name=True)
        model = self.models[best_name]

        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Feature importance not supported for {best_name}")
            return

        names = X_test.columns.tolist()
        df_imp = pd.DataFrame({
            'Feature': names,
            'Gain_Importance': model.feature_importances_
        })

        # XGBoost was fitted on encoded labels, so align y_test accordingly
        y_eval = y_test
        if best_name == 'xgboost' and self.label_encoder is not None:
            y_eval = pd.Series(self.label_encoder.transform(y_test), index=y_test.index)

        perm = permutation_importance(
            model, X_test, y_eval, n_repeats=5, random_state=self.seed,
            n_jobs=-1, scoring='f1_weighted'
        )
        df_imp['Permutation_Importance'] = perm.importances_mean
        df_imp = df_imp.sort_values('Gain_Importance', ascending=False)

        path = Path(output_dir) / "feature_importance.csv"
        df_imp.to_csv(path, index=False)
        logger.info(f"Saved feature importance to {path}")


if __name__ == "__main__":
    from config import PROCESSED_DIR, MODELS_DIR

    processed_path = PROCESSED_DIR / "processed_data.parquet"
    models_dir = MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    print("ML Model Training Pipeline")
    print("=" * 50)

    if not processed_path.exists():
        raise SystemExit(f"Processed data not found: {processed_path}. Run data_processor.py first.")

    df = pd.read_parquet(processed_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"Loaded processed data: {df.shape}")

    trainer = MLModelTrainer(seed=42)

    # Prepare features/target (original labels {-1,0,1})
    X_train, X_test, y_train, y_test, encoders = trainer.prepare_data(df, target_col='target')
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Train multiple models (use a light subset to keep runtime reasonable)
    models_to_train = ['decision_tree', 'random_forest', 'xgboost', 'lightgbm']
    trainer.train_all_models(X_train, X_test, y_train, y_test,
                             models_to_train=models_to_train, cv_splits=3)

    # Show performance table
    perf = trainer.get_model_performance_table()
    print("\nModel Performance:")
    print(perf)

    # Save models and metrics
    trainer.save_models(str(models_dir))

    # Select best model and generate predictions on the full dataset
    best_name = trainer.select_best_model(return_name=True)
    print(f"\nBest model: {best_name}")

    trainer.save_feature_importance(X_test, y_test, str(models_dir))

    # Build the full feature matrix (same columns as training)
    feature_cols = X_train.columns.tolist()
    X_all = df[feature_cols].ffill().bfill().reset_index(drop=True)

    pred_class, prob_up, prob_down = trainer.predict_full(best_name, X_all)

    preds = df[['symbol', 'date', 'Close', 'target']].copy().reset_index(drop=True)
    preds['pred_class'] = pred_class
    preds['prob_up'] = prob_up
    preds['prob_down'] = prob_down
    if 'BENCHMARK_returns' in df.columns:
        preds['BENCHMARK_returns'] = df['BENCHMARK_returns'].values

    preds_path = PROCESSED_DIR / "predictions.parquet"
    preds.to_parquet(preds_path)
    print(f"Saved predictions to {preds_path}")

    # Save best model name
    import json
    with open(models_dir / "best_model.json", "w") as f:
        json.dump({'best_model': best_name}, f, indent=2)

