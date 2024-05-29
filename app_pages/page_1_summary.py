import streamlit as st

def page_1_summary_body():
    """
    Displays a quick project summary, including dataset content, business requirements,
    and links to further information.
    """
    st.write("### Quick Project Summary")

    # Text based on README file - "Dataset Content" section
    st.info(
        """
        **Project Terms & Jargons**
        * **SalePrice** is the price a house sold for and is our target variable.

        **Project Dataset**
        * The dataset represents housing records from Ames, Iowa, indicating house profile 
        (Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and its 
        respective sale price for houses built between 1872 and 2010.
        * There are many abbreviated terms used to describe features of the houses in 
        the data set. For further clarification of the full dataset and explanation
        of its terms, you can click **[HERE](https://www.kaggle.com/datasets/codeinstitute/housing-prices-data)**.
        """
    )

    # Link to README file, so the users can have access to full project documentation
    st.write(
        """
        For additional information, please visit and **read** the 
        **[Project's README file](https://github.com/Tomosius/heritage_houses_p5)**.
        """
    )

    # Copied from README file - "Business Requirements" section
    st.success(
        """
        #### The project has two business requirements:

        **1.** The client is interested in discovering how house attributes correlate with 
        the house Sale Price. Therefore, the client expects data visualizations 
        of the correlated variables against Sale Price to show that.

        **2.** The client is interested to predict the house sales price from their 4 
        inherited houses, and any other house in Ames, Iowa.
        """
    )

    # Overview of CRISP-DM Methodology
    st.write("### CRISP-DM Methodology")
    st.markdown(
        """
        CRISP-DM (Cross-Industry Standard Process for Data Mining) is a widely-used methodology 
        that provides a structured approach to planning and executing data mining and machine 
        learning projects. The methodology is divided into six phases:

        1. **Business Understanding**: Understanding the project objectives and requirements.
        2. **Data Understanding**: Initial data collection and exploration.
        3. **Data Preparation**: Data cleaning and preparation for modeling.
        4. **Modeling**: Selecting and applying various modeling techniques.
        5. **Evaluation**: Evaluating the models to ensure they meet the business requirements.
        6. **Deployment**: Deploying the model for practical use.
        """
    )

    # Summary of the "Business Requirements" section from the README
    st.write("### Business Requirements")
    st.markdown(
        """
        The project was initiated by a local heritage society aiming to maximize the preservation 
        and restoration value of historical houses in Iowa. The society needs predictive analytics 
        to accurately appraise the historical value and restoration costs of these properties.

        **Business Objectives:**

        1. **Understanding House Attributes and Historical Value:**
           - Discover how different attributes of historical houses correlate with their estimated historical value.
           - Provide data visualizations to depict the relationship between house attributes and historical value.

        2. **Predicting Restoration Costs and Historical Value:**
           - Predict the potential historical value for several historical houses.
           - Guide decisions about the buildings' future based on these predictions.

        **Specific Requirements:**

        - **Data Visualization:**
          - Create visualizations illustrating the correlation between various house attributes and their historical value.
          - Use scatter plots, bar charts, and heatmaps for clear insights.

        - **Predictive Modeling:**
          - Develop machine learning models to predict the historical value based on the attributes of the houses.
          - Ensure models are trained and validated on the provided dataset for high accuracy and reliability.
        """
    )

