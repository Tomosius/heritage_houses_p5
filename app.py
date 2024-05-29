import streamlit as st
from app_pages.multipage import MultiPage

import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_1_summary import page_1_summary_body


app = MultiPage(app_name= "Heritage Housing") # Create an instance of the app

# Add your app pages here using .add_page()
app.app_page("Project Summary", page_1_summary_body)


app.run() # Run the  app

# code copied from Code Institute's Churnornmeter Project with some adjustments