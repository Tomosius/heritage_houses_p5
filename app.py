import app_pages.page_5_ml_prediction as ml_prediction
from app_pages.multipage import MultiPage
# Load page scripts
from app_pages.page_1_summary import page_1_summary_body
from app_pages.page_2_study_sale_price import page_sale_price_study_body
from app_pages.page_4_hypothesis_and_validation import hypothesis_3_analysis
from app_pages.page_6_bulk_predict import page_sale_price_predictions_body

# Create an instance of the app
app = MultiPage(app_name="Heritage Housing")

# Add your app pages here using .app_page()
app.app_page("Project Summary", page_1_summary_body)
app.app_page("Features and Correlations", page_sale_price_study_body)
app.app_page("Target Predictions", page_sale_price_predictions_body)
app.app_page("Hypothesis and Validation", hypothesis_3_analysis)
app.app_page("ML Model", ml_prediction.display)
app.app_page("Bulk Predictions", page_sale_price_predictions_body)

# Run the app
app.run()
