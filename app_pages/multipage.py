import streamlit as st

class MultiPage:
    def __init__(self, app_name) -> None:
        self.pages = []
        self.app_name = app_name

        st.set_page_config(
            page_title=self.app_name,
            page_icon="🖥️")  # You may add an icon, to personalize your App

    def app_page(self, title, func) -> None:
        """
        Add a new page to the app.

        Args:
            title (str): The title of the page.
            func (function): The function to render the page.
        """
        self.pages.append({"title": title, "function": func })

    def run(self):
        """
        Run the app and display the selected page.
        """
        st.title(self.app_name)
        page = st.sidebar.radio('Menu', self.pages, format_func=lambda page: page['title'])
        page['function']()
