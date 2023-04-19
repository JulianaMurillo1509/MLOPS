import requests
import streamlit as st
import pandas as pd


STYLES = {
    "candy": "candy",
    "composition 6": "composition_vii",
    "feathers": "feathers",
    "la_muse": "la_muse",
    "mosaic": "mosaic",
    "starry night": "starry_night",
    "the scream": "the_scream",
    "the wave": "the_wave",
    "udnie": "udnie",
}

# https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("Proyecto 2 - Cover type web app")

# displays a file uploader widget
file = st.file_uploader(  "",
        key="1",
        help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",)

# displays the select widget for the styles
#style = st.selectbox("Choose the style", [i for i in STYLES.keys()])

# displays a button
if st.button("train"):
    if file is not None:
        file_container = st.expander("Check your uploaded .csv")
        shows = pd.read_csv(file)
        file.seek(0)
        file_container.write(shows)

    else:
        st.info(
            f"""
                👆 Upload a .csv file first.
                """
        )
