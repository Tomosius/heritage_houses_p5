import streamlit as st
from app_pages.multipage import MultiPage

import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_1_summary import page_1_summary_body
from app_pages.page_2_study_sale_price import page_sale_price_study_body
from app_pages.page_3_sale_price_predictions import page_sale_price_predictions_body
from app_pages.page_4_hypothesis_and_validation import hypothesis_3_analysis

app = MultiPage(app_name="Heritage Housing")  # Create an instance of the app

# Add your app pages here using .add_page()
app.app_page("Project Summary", page_1_summary_body)
app.app_page("Features and correlations", page_sale_price_study_body)
app.app_page("Target Predictions", page_sale_price_predictions_body)
app.app_page("Hypothesis and validation", hypothesis_3_analysis)

app.run()  # Run the  app
