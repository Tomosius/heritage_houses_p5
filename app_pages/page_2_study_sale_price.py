import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from file_management import load_house_data, load_pkl_file

sns.set_style("whitegrid")


def page_sale_price_study_body():
    df = load_house_data()
    corr_var_list = ['YearBuilt', 'GarageYrBlt', 'LotArea', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'OverallQual',
                     'OverallCond']

    st.write("### House Sale Price Study")
    st.info(
        "The local heritage society is interested in discovering how the attributes "
        "of historical houses correlate with their sale price. This information will help "
        "the society maximize the preservation and restoration value of these culturally "
        "significant properties in Iowa.\n"
        "* Therefore, the society expects data visualizations of the "
        "correlated variables against the sale price to provide clear insights into these relationships."
    )

    if st.checkbox("Inspect House Price Data:"):
        st.write(
            f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns. The first 10 rows are displayed below.")
        st.write(df.head(10))

    st.write("---")

    st.success(
        "Correlation studies were conducted using the Pearson and Spearman methods to better understand how the "
        "variables correlate to the"
        "sale price.\n"
        "ML model was built on these features: \n"
        "* **YearBuilt, GarageYrBlt, LotArea, NF_1stFlrSF_mul_OverallQual, NF_TotalLivingArea_mul_OverallCond, "
        "NF_TotalLivingArea_mul_OverallQual**"
    )

    st.info(
        "### The correlations and plots interpretation converge.\n"
        "The following are the variables isolated in the correlation study:\n"
        "* **YearBuilt:** Original construction date (1872 to 2010).\n"
        "* **GarageYrBlt:** Garage built date (1900 to 2010).\n"
        "* **LotArea:** Lot Size in square feet (1300 - 215245).\n"
        "* Also for building model new sub_features (prefix NF_) were created: \n"
        "* **NF_1stFlrSF_mul_OverallQual:** 1st floor area in square feet multiplied by Overall material and finish "
        "of the house \n"
        "* **NF_TotalLivingArea_mul_OverallCond:** Total living area in square feet multiplied by overall condition "
        "rate of the house \n"
        "* **NF_TotalLivingArea:** This feature represents the total living area of the house by summing the ground "
        "living area (GrLivArea), the first-floor area (1stFlrSF), and the second-floor area (2ndFlrSF). This "
        "provides a comprehensive measure of the house's total living space, which is likely to be a significant "
        "factor in determining its sale price."
        "* **NF_TotalLivingArea_mul_OverallCond and NF_TotalLivingArea_mul_OverallQual:** These are also new "
        "features, made by multiplying **NF_TotalLivingArea** by: Overall material & finish of house and Overall "
        "condition of the house accordingly"
    )

    df_filtered = df[corr_var_list].copy()
    df_target = df['SalePrice'].copy()

    model_pipeline = load_pkl_file()
    pre_feature_transformations = model_pipeline.named_steps['pre_transformations']
    df_filtered_transformed = pre_feature_transformations.fit_transform(df_filtered)

    st.write("### Feature Correlations with Sale Price")
    plot_features_vs_target_correlations(df_filtered_transformed, df_target)

    st.write("### Feature Distributions vs Sale Price")
    plot_features_vs_target_distribution(df_filtered_transformed, df_target)


def plot_features_vs_target_distribution(dataset, target):
    numerical_features = dataset.columns.tolist()

    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=dataset[feature], y=target)
        plt.title(f'SalePrice vs {feature}')
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        st.pyplot(plt)
        plt.clf()


def plot_features_vs_target_correlations(dataset, target):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Compute the Pearson correlation of each feature with the target variable
    pearson_corr = dataset.apply(lambda x: x.corr(target, method='pearson'))

    # Compute the Spearman correlation of each feature with the target variable
    spearman_corr = dataset.apply(lambda x: x.corr(target, method='spearman'))

    # Combine the correlations into a DataFrame
    correlation_df = pd.DataFrame({
        'Pearson': pearson_corr,
        'Spearman': spearman_corr
    })

    # Plot the combined correlations
    correlation_df_sorted = correlation_df.sort_values(by='Pearson', ascending=False)

    # Create a grouped bar plot
    plt.figure(figsize=(14, 8))
    bar_width = 0.35
    index = np.arange(len(correlation_df_sorted))

    bars1 = plt.bar(index, correlation_df_sorted['Pearson'], bar_width, label='Pearson', color='skyblue')
    bars2 = plt.bar(index + bar_width, correlation_df_sorted['Spearman'], bar_width, label='Spearman',
                    color='lightgreen')

    plt.xlabel('Features')
    plt.ylabel('Correlation Coefficient')
    plt.title('Pearson and Spearman Correlation of Features with SalePrice')
    plt.xticks(index + bar_width / 2, correlation_df_sorted.index, rotation=45, ha='right')
    plt.legend()

    # Add the correlation coefficients on top of each bar
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom', rotation=90)

    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom', rotation=90)

    st.pyplot(plt)  # Ensure the plot is displayed in Streamlit


if __name__ == "__main__":
    page_sale_price_study_body()
