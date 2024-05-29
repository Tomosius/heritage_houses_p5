import seaborn as sns
import streamlit as st
import pandas as pd
from custom_transformers import DatasetCleaner
from file_management import load_house_data, load_pkl_file

# Set seaborn style for plots
sns.set_style("whitegrid")

# Constants
MIN_YEAR = 1872
MAX_YEAR = 2010

# Mapping for OverallCond and OverallQual ratings
rating_text = {
    10: "Very Excellent",
    9: "Excellent",
    8: "Very Good",
    7: "Good",
    6: "Above Average",
    5: "Average",
    4: "Below Average",
    3: "Fair",
    2: "Poor",
    1: "Very Poor",
}


def page_sale_price_predictions_body():
    """
    Renders the main body of the house sale price predictions page.
    """
    # List of variables used for correlation analysis and predictions
    corr_var_list = ['YearBuilt', 'GarageYrBlt', 'LotArea', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'OverallQual',
                     'OverallCond']

    st.write("### House Sale Price Predictions")
    st.info(
        "The society wants to predict the potential historical value for several historical houses, including their "
        "own properties and other significant houses in **Iowa**."
        "These predictions will help determine their value and guide decisions about the buildings' future."
    )

    # Call function to create input data widgets
    input_data = input_data_widgets(corr_var_list)
    model_pipeline = load_pkl_file()

    # Automatically predict based on current input data
    try:
        auto_prediction = model_pipeline.predict(input_data)
        # Display the automatic prediction as a single value
        st.write(f"Based on current data input, the predicted SalePrice is ${auto_prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error making automatic prediction: {e}")


def input_data_widgets(corr_var_list):
    """
    Creates and displays input widgets for house data attributes.

    Args:
        corr_var_list (list): List of features to keep for correlation analysis.

    Returns:
        pd.DataFrame: DataFrame containing the input data from the user.
    """
    try:
        # Load the dataset
        df = load_house_data()

        # Keep only the specified features
        df = df[corr_var_list]

        # Apply DatasetCleaner to clean the data
        cleaner = DatasetCleaner()
        df = cleaner.fit_transform(df)

        # Convert all float columns to int
        df = df.apply(lambda x: x.astype(int) if x.dtype == 'float64' else x)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

    # Create columns for widgets layout
    col1, col2, col3 = st.beta_columns(3)
    col4, col5, col6 = st.beta_columns(3)
    col7, col8 = st.beta_columns(2)

    # Initialize an empty DataFrame for input data
    input_data = pd.DataFrame([], index=[0])

    # Dictionary to map columns to their respective features
    columns = {
        col1: "1stFlrSF",
        col2: "2ndFlrSF",
        col3: "GrLivArea",
        col4: "YearBuilt",
        col5: "GarageYrBlt",
        col6: "LotArea",
        col7: "OverallCond",
        col8: "OverallQual"
    }

    # Initialize placeholders for warnings
    garage_warning_placeholder = st.empty()
    year_built_warning_placeholder = st.empty()
    floor_area_warning_placeholder = st.empty()
    negative_value_warning_placeholder = st.empty()

    # Generate input widgets for each feature
    for col, feature in columns.items():
        with col:
            if feature in ["OverallCond", "OverallQual"]:
                options = [f"{rating_text[i]} ({i})" for i in rating_text]
                st_widget = st.selectbox(
                    label=f"Please select {feature}",
                    options=options,
                    key=f"{feature}_selectbox"  # Ensuring a unique key for each selectbox
                )
                selected_value = int(st_widget.split('(')[-1].strip(')'))
                input_data[feature] = selected_value
            else:
                st_widget = st.number_input(
                    label=feature,
                    value=int(df[feature].mean()),
                    step=1,
                    min_value=0,  # Ensure no negative values
                    key=f"{feature}_numberinput"  # Ensuring a unique key for each number input
                )
                input_data[feature] = st_widget

            # Additional handling for specific features
            if feature == "YearBuilt":
                handle_year_built_warning(st_widget, year_built_warning_placeholder)
            elif feature == "GarageYrBlt":
                handle_garage_year_warning(st_widget, input_data, garage_warning_placeholder)

    # Ensure all input data is numeric
    input_data = input_data.apply(pd.to_numeric, errors='coerce')

    # Add additional checks for floor areas and negative values
    handle_floor_area_warnings(input_data, floor_area_warning_placeholder)
    handle_negative_value_warnings(input_data, negative_value_warning_placeholder)

    return input_data


def handle_year_built_warning(year_built, placeholder):
    """
    Handles warnings for the YearBuilt feature based on its value.

    Args:
        year_built (int): The input year built value.
        placeholder (Streamlit component): The placeholder for displaying warnings.
    """
    if year_built < MIN_YEAR or year_built > MAX_YEAR:
        placeholder.warning(
            f"Model was trained on values for Building Year between {MIN_YEAR} and {MAX_YEAR}. "
            "Using values outside this range may result in inaccurate predictions."
        )
    else:
        placeholder.empty()


def handle_garage_year_warning(garage_year, input_data, placeholder):
    """
    Handles warnings for the GarageYrBlt feature based on its value.

    Args:
        garage_year (int): The input garage build year value.
        input_data (pd.DataFrame): DataFrame containing the input data from the user.
        placeholder (Streamlit component): The placeholder for displaying warnings.
    """
    if garage_year < MIN_YEAR or garage_year > MAX_YEAR:
        placeholder.warning(
            f"Model was trained on values for Garage Build Year between {MIN_YEAR} and {MAX_YEAR}. "
            "Using values outside this range may result in inaccurate predictions."
        )
    elif 'YearBuilt' in input_data and garage_year < input_data['YearBuilt'].values[0]:
        placeholder.warning(
            "The garage build year cannot be earlier than the building year. Please correct it."
        )
    else:
        placeholder.empty()


def handle_floor_area_warnings(input_data, placeholder):
    """
    Handles warnings for floor area features based on their values.

    Args:
        input_data (pd.DataFrame): DataFrame containing the input data from the user.
        placeholder (Streamlit component): The placeholder for displaying warnings.
    """
    first_floor_sf = input_data['1stFlrSF'].values[0]
    second_floor_sf = input_data['2ndFlrSF'].values[0]
    gr_liv_area = input_data['GrLivArea'].values[0]

    if second_floor_sf > first_floor_sf:
        placeholder.warning("2ndFlrSF must be smaller or equal to 1stFlrSF.")
    elif first_floor_sf > gr_liv_area:
        placeholder.warning("1stFlrSF must be smaller or equal to GrLivArea.")
    else:
        placeholder.empty()


def handle_negative_value_warnings(input_data, placeholder):
    """
    Handles warnings for negative values in the input data.

    Args:
        input_data (pd.DataFrame): DataFrame containing the input data from the user.
        placeholder (Streamlit component): The placeholder for displaying warnings.
    """
    negative_values = input_data[input_data < 0].dropna(axis=1)
    if not negative_values.empty:
        placeholder.warning("All values must be non-negative.")
    else:
        placeholder.empty()


if __name__ == "__main__":
    page_sale_price_predictions_body()
