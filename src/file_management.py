import streamlit as st
import pandas as pd
import joblib

# Define constants for file paths
HOUSE_PRICES_CSV_PATH = "outputs/datasets/collection/HousePricesRecords.csv"
INHERITED_HOUSES_CSV_PATH = "outputs/datasets/collection/InheritedHouses.csv"
MODEL_PKL_PATH = 'outputs/ml_pipeline/predict_sale_price.pkl'


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_house_data():
    """
    Load the house prices dataset from a CSV file.

    Returns:
        pd.DataFrame: The house prices dataset.
    """
    try:
        dataset = pd.read_csv(HOUSE_PRICES_CSV_PATH)
        return dataset
    except Exception as e:
        st.error(f"Error loading house prices dataset: {e}")
        return pd.DataFrame()


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_inherited_house_data():
    """
    Load the inherited houses dataset from a CSV file.

    Returns:
        pd.DataFrame: The inherited houses dataset.
    """
    try:
        inherited = pd.read_csv(INHERITED_HOUSES_CSV_PATH)
        return inherited
    except Exception as e:
        st.error(f"Error loading inherited houses dataset: {e}")
        return pd.DataFrame()


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_pkl_file():
    """
    Load a pickled machine learning model.

    Returns:
        The loaded machine learning model.
    """
    try:
        model = joblib.load(MODEL_PKL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None
