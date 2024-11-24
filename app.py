import streamlit as st
from video_process_page import video_process_page
from home_page import home_page  # Import the home page
from streamlit_navigation_bar import st_navbar

def main():
    page = st_navbar(["Home", "Documentation", "Examples", "Community", "About"])
    st.write(page)

if __name__ == "__main__":
    main()