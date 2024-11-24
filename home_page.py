import streamlit as st

def home_page():
    st.title("Welcome to Car Crash Detection and Tracking App")
    st.write("This is a Streamlit app using to represent car crash detection and tracking model.")
    st.write("Here is the example result of the model")
    
    # Replace the image with a video player
    video_url = "https://www.w3schools.com/html/mov_bbb.mp4"  # You can replace this with your own video URL
    st.video(video_url)

    st.markdown(
        """
        ### Features of the Model:
        - Navigate between pages using the sidebar.
        - Process videos on the "Video Processing" page.
        - Explore more features coming soon!
        """
    )
