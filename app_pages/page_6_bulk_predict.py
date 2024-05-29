import seaborn as sns
import streamlit as st
import pandas as pd
from file_management import load_pkl_file, load_inherited_house_data

# Set seaborn style for plots
sns.set_style("whitegrid")

# Define the columns used for predictions
PREDICTION_COLUMNS = ['YearBuilt', 'GarageYrBlt', 'LotArea', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'OverallQual', 'OverallCond']

def page_sale_price_predictions_body_bulk():
    """
    Renders the main body of the house sale price predictions page.
    """
    st.write("### House Sale Price Predictions")
    st.info(
        "The society wants to predict the potential historical value for several historical houses, including their "
        "own properties and other significant houses in **Iowa**."
        "These predictions will help determine their value and guide decisions about the buildings' future."
    )

    # Load initial inherited house data and make predictions
    inherited_data = load_inherited_house_data()
    st.write("#### Initial Data from Inherited House Dataset")
    st.dataframe(inherited_data)

    processed_data = process_data(inherited_data)
    st.write("#### Predictions for Inherited Data")
    st.dataframe(processed_data)

    # Provide file uploader for user to upload CSV
    uploaded_file = st.file_uploader("Upload your CSV file for house data", type=["csv"])

    if uploaded_file:
        try:
            user_data = pd.read_csv(uploaded_file)
            st.write("#### Uploaded Data")
            st.dataframe(user_data)

            processed_user_data = process_data(user_data)
            st.write("#### Predictions for Uploaded Data")
            st.dataframe(processed_user_data)
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")

def process_data(data):
    """
    Processes the dataset to prepare it for predictions.

    Args:
        data (pd.DataFrame): DataFrame containing the house data.

    Returns:
        pd.DataFrame: DataFrame with predictions.
    """
    model_pipeline = load_pkl_file()

    # Select relevant columns
    df_predict = data[PREDICTION_COLUMNS]

    # Make predictions
    predictions = model_pipeline.predict(df_predict)
    df_predict.loc[:, 'SalePrice'] = predictions.round().astype(int)

    return df_predict

if __name__ == "__main__":
    page_sale_price_predictions_body()
