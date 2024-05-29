import streamlit as st
from app_pages.multipage import MultiPage

import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_1_summary import page_1_summary_body
from app_pages.page_2_study_sale_price import page_sale_price_study_body


app = MultiPage(app_name= "Heritage Housing") # Create an instance of the app

# Add your app pages here using .add_page()
app.app_page("Project Summary", page_1_summary_body)
app.app_page("Project Sale", page_sale_price_study_body)


app.run() # Run the  app

