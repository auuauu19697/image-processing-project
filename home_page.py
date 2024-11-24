import streamlit as st

def home_page():
    st.title("Welcome to the Home Page!")
    st.write("This is the home page of the app.")
    st.image("https://via.placeholder.com/800x400", caption="A placeholder image for your home page")
    st.markdown(
        """
        ### Features of the App:
        - Navigate between pages using the sidebar.
        - Process videos on the "Video Processing" page.
        - Explore more features coming soon!
        """
    )
