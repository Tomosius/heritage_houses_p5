import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from feature_engine.transformation import YeoJohnsonTransformer, PowerTransformer
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import joblib

# Define custom transformers
class DatasetCleaner(BaseEstimator, TransformerMixin):
    """Custom dataset cleaner for pipeline integration."""

    def __init__(self):
        self.cleaning_descriptions_ = {}
        self.output_features_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        fill_zero_and_convert = ['1stFlrSF', '2ndFlrSF', 'GarageYrBlt']
        X[fill_zero_and_convert] = X[fill_zero_and_convert].fillna(0).astype(int)
        self.cleaning_descriptions_['fill_zero_and_convert'] = (
            f"Filled missing values in {fill_zero_and_convert} with 0 and converted to int."
        )

        swap_idx = X['2ndFlrSF'] > X['1stFlrSF']
        X.loc[swap_idx, ['1stFlrSF', '2ndFlrSF']] = X.loc[swap_idx, ['2ndFlrSF', '1stFlrSF']].values
        self.cleaning_descriptions_['swap_flrsf'] = "Swapped values where '2ndFlrSF' is greater than '1stFlrSF'."

        X.loc[X['GarageYrBlt'] < X['YearBuilt'], 'GarageYrBlt'] = X['YearBuilt']
        self.cleaning_descriptions_['correct_garage_year'] = "Corrected garage years that are earlier than the house build year."

        return X

    def get_cleaning_descriptions(self):
        return self.cleaning_descriptions_

    def get_feature_names_out(self, input_features=None):
        return self.output_features_

    def __repr__(self):
        cleaning_descriptions = "\n".join(
            [f"{step}: {desc}" for step, desc in self.cleaning_descriptions_.items()])
        return f"DatasetCleaner(cleaning_steps:\n{cleaning_descriptions})"

class FeatureCreator(BaseEstimator, TransformerMixin):
    """Custom feature creator for pipeline integration."""

    def __init__(self):
        self.feature_creation_descriptions_ = {}
        self.output_features_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        self.feature_creation_descriptions_ = {
            'NF_TotalLivingArea': 'GrLivArea + 1stFlrSF + 2ndFlrSF',
            'NF_TotalLivingArea_mul_OverallQual': 'NF_TotalLivingArea * OverallQual',
            'NF_TotalLivingArea_mul_OverallCond': 'NF_TotalLivingArea * OverallCond',
            'NF_1stFlrSF_mul_OverallQual': '1stFlrSF * OverallQual',
        }

        X['NF_TotalLivingArea'] = X['GrLivArea'] + X['1stFlrSF'] + X['2ndFlrSF']
        X['NF_TotalLivingArea_mul_OverallQual'] = X['NF_TotalLivingArea'] * X['OverallQual']
        X['NF_TotalLivingArea_mul_OverallCond'] = X['NF_TotalLivingArea'] * X['OverallCond']
        X['NF_1stFlrSF_mul_OverallQual'] = X['1stFlrSF'] * X['OverallQual']

        necessary_features = ['YearBuilt', 'GarageYrBlt', 'LotArea', 'NF_1stFlrSF_mul_OverallQual',
                              'NF_TotalLivingArea_mul_OverallCond', 'NF_TotalLivingArea_mul_OverallQual']
        X = X[necessary_features]

        self.output_features_ = X.columns.tolist()
        return X

    def get_feature_names_out(self, input_features=None):
        return self.output_features_

    def __repr__(self):
        feature_descriptions = "\n".join(
            [f"{name}: {desc}" for name, desc in self.feature_creation_descriptions_.items()])
        return f"FeatureCreator(created_features:\n{feature_descriptions})"


# Pipelines
pre_feature_transformations = Pipeline([
    ('dataset_cleaner', DatasetCleaner()),
    ('feature_creator', FeatureCreator())
])


from sklearn.pipeline import Pipeline
from feature_engine.transformation import YeoJohnsonTransformer, PowerTransformer

# Define the columns for each transformation type
yeo_johnson_features = ['LotArea', 'NF_TotalLivingArea_mul_OverallQual', 'NF_TotalLivingArea_mul_OverallCond',
                        'NF_1stFlrSF_mul_OverallQual']
power_features = ['GarageYrBlt']

# Create transformers for each group of features using feature_engine transformers
yeo_johnson_transformer = YeoJohnsonTransformer(variables=yeo_johnson_features)
power_transformer = PowerTransformer(variables=power_features, exp=0.5)

# Combine all transformers into a single pipeline
feature_transformer = Pipeline([
    ('yeo_johnson', yeo_johnson_transformer),
    ('power', power_transformer),
])


from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

# Define the columns for Winsorization
winsorize_features = ['LotArea', 'NF_TotalLivingArea_mul_OverallCond']

# Initialize the Winsorizer transformer
# We will apply Winsorizer to features from table in jupyter_notebooks/08_Feature_Engineering_hypothesis_3.ipynb
# The ones which gad high or above outliers
winsorize_transformer = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=winsorize_features)

# Create the post-feature transformations pipeline
post_feature_transformer = Pipeline([
    ('winsorize', winsorize_transformer),
    ('standard_scaler', StandardScaler())
])


from sklearn.base import BaseEstimator, TransformerMixin


class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Add a small constant to avoid log(0)
        return np.log1p(np.clip(X, 0, None))

    def inverse_transform(self, X):
        # Use expm1 for numerical stability, clip to avoid overflow
        return np.expm1(np.clip(X, None, 700))  # 700 is chosen to avoid overflow in expm1


# Create a pipeline for transforming the target variable
target_transformation_pipeline = Pipeline([
    ('log_transform', LogTransformer()),  # Log transformation
])

