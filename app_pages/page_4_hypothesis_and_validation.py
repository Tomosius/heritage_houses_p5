import streamlit as st


def display_title():
    """Display the title for the hypothesis analysis."""
    st.title("Hypothesis 3 Analysis and Validation")


def display_new_features():
    """Display the new features and their explanations."""
    st.header("New Features (NF) Calculations and Explanations")
    st.write("Here we describe the new features created from the existing dataset and their significance in the model.")

    new_features = {
        "NF_TotalLivingArea": {
            "calculation": "NF_TotalLivingArea = GrLivArea + 1stFlrSF + 2ndFlrSF",
            "explanation": "This feature represents the total living area of the house by summing the ground living "
                           "area (GrLivArea), the first-floor area (1stFlrSF), and the second-floor area (2ndFlrSF). "
                           "It provides a comprehensive measure of the house's total living space, which likely "
                           "influences its sale price."
        },
        "NF_TotalLivingArea_mul_OverallQual": {
            "calculation": "NF_TotalLivingArea_mul_OverallQual = NF_TotalLivingArea * OverallQual",
            "explanation": "This feature is calculated by multiplying the total living area (NF_TotalLivingArea) by "
                           "the overall quality of the house (OverallQual). It helps incorporate the quality of the "
                           "living space into the model, as higher quality construction and finishes typically "
                           "increase the house's value."
        },
        "NF_TotalLivingArea_mul_OverallCond": {
            "calculation": "NF_TotalLivingArea_mul_OverallCond = NF_TotalLivingArea * OverallCond",
            "explanation": "This feature is calculated by multiplying the total living area (NF_TotalLivingArea) by "
                           "the overall condition of the house (OverallCond). It accounts for the condition of the "
                           "living space, with better condition typically leading to a higher sale price."
        },
        "NF_1stFlrSF_mul_OverallQual": {
            "calculation": "NF_1stFlrSF_mul_OverallQual = 1stFlrSF * OverallQual",
            "explanation": "This feature is calculated by multiplying the first-floor square feet (1stFlrSF) by the "
                           "overall quality of the house (OverallQual). It emphasizes the importance of the quality "
                           "of the main living area on the first floor, which can significantly impact the house's "
                           "overall value."
        }
    }

    for feature, details in new_features.items():
        st.subheader(feature)
        st.markdown(f"""
        - **Calculation**:
          ```
          {details['calculation']}
          ```
        - **Explanation**: {details['explanation']}
        """)


def display_selected_features_correlation():
    """Display the correlation between selected features and sale price."""
    st.header("Selected Features and their Correlation to Sale Price")
    st.write(
        "The correlation between selected features and the sale price of the house is illustrated in the chart below.")
    try:
        st.image("src/images/hypothesis_3_features_correlations_vs_sale_price.png",
                 caption="Correlation of Selected Features to Sale Price")
    except Exception as e:
        st.error(f"Error loading image: {e}")


def display_feature_distributions():
    """Display the distribution of selected features against sale price."""
    st.header("Selected Features Distribution Against Sale Price")
    feature_images = {
        "sale_price_vs_NF_TotalLivingArea_mul_OverallQual.png": "Sale Price vs NF_TotalLivingArea_mul_OverallQual",
        "sale_price_vs_NF_TotalLivingArea_mul_OverallCond.png": "Sale Price vs NF_TotalLivingArea_mul_OverallCond",
        "sale_orice_vs_NF_1stFlrSF_mul_OverallQual.png": "Sale Price vs NF_1stFlrSF_mul_OverallQual",
        "sale_price_vs_lot_area.png": "Sale Price vs Lot Area",
        "saleprice_vs_garageyearbuilt.png": "Sale Price vs Garage Year Built",
        "saleprice_vs_yearbuilt.png": "Sale Price vs Year Built"
    }

    feature_descriptions = {
        "sale_price_vs_NF_TotalLivingArea_mul_OverallQual.png": "We can see that when "
                                                                "NF_TotalLivingArea_mul_OverallQual increases "
                                                                "SalePrice also increases and it is very highly "
                                                                "correlated",
        "sale_price_vs_NF_TotalLivingArea_mul_OverallCond.png": "On this plot We can explore similar correlation "
                                                                "between NF_TotalLivingArea_mul_OverallCond and "
                                                                "SalePrice, except with an increase of "
                                                                "NF_TotalLivingArea_mul_OverallCond sale",
        "sale_orice_vs_NF_1stFlrSF_mul_OverallQual.png": "This plot displays again very similar distribution between "
                                                         "NF_1stFlrSF_mul_OverallQual and SalePrice, just a bit less "
                                                         "scattered when NF_1stFlrSF_mul_OverallQual increases",
        "sale_price_vs_lot_area.png": "Lot Area is very highly correlated to SalePrice, as it increases SalePrice "
                                      "also goes up",
        "saleprice_vs_garageyearbuilt.png": "Garage Year Built is also highly correlated to SalePrice and with later "
                                            "Build date SalePrice goes up",
        "saleprice_vs_yearbuilt.png": "Very similar pattern to Garage Year Built, The later construction date - the "
                                      "higher the price"
    }

    for image, caption in feature_images.items():
        try:
            st.image(f"src/images/{image}", caption=caption)
            st.write(feature_descriptions[image])
        except Exception as e:
            st.error(f"Error loading image {image}: {e}")


def display_hypothesis_validation():
    """Display the validation of Hypothesis 3."""
    st.header("Hypothesis 3 Validation")
    st.write("""After exploring new features, we gained more insight into the dataset. The result improved to 0.843, 
    validating our hypothesis.""")
    try:
        st.image("src/images/hypothesis_3_validation.png", caption="Hypothesis 3 Model Evaluation")
    except Exception as e:
        st.error(f"Error loading image: {e}")


def hypothesis_3_analysis():
    """Main function to run the hypothesis 3 analysis and validation app."""
    display_title()
    display_new_features()
    display_selected_features_correlation()
    display_feature_distributions()
    display_hypothesis_validation()


if __name__ == "__main__":
    hypothesis_3_analysis()
