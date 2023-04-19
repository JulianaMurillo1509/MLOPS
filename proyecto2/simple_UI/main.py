import requests
import streamlit as st
import pandas as pd


ISLAND = {
    "Torgersen": "Torgersen",
    "Biscoe": "Biscoe",
    "Dream": "Dream",
}


SEX = {
    "male": "male",
    "female": "female",
}


URL_TRAIN = 'http://localhost:8000/train_model/uploadfile/insertDatainDB'

# https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("Proyecto 2 - Cover type web app")

# displays a file uploader widget
st.header('Section to upload file')
file = st.file_uploader(  "",
        key="1",
        help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",)
# displays a button

if st.button("Upload file"):
    if file is not None:
        files = {"file": file.getvalue()}
        response = requests.post("http://api_train:80/train_model/uploadfile/", files=files)
        st.write(response.json())
    else:
        st.write("Please choose a file to upload.")
st.header('Section to Train Model')
if st.button("train model"):
        response = requests.get("http://api_train:80/train_model/train/")
        st.write(response)


st.header('Section to do Inference')
with st.form("my_form"):
    island = st.selectbox("Choose the island", [i for i in ISLAND.keys()])
    sex = st.selectbox("Choose the penguin sex", [i for i in SEX.keys()])
    billlengthmm = st.text_input('text1')
    billdepthmm = st.text_input('text2')
    flipperlengthmm = st.text_input('text3')
    bodymassg = st.text_input('text4')
    year = st.text_input('text5')
    submitted = st.form_submit_button("Submit")
    if submitted:
        data = {
            "island": island,
            "sex": sex,
            "bill_length_mm": billlengthmm,
            "bill_depth_mm": billdepthmm,
            "flipper_length_mm": flipperlengthmm,
            "body_mass_g": bodymassg,
            "year": year
            }
            
        response = requests.post("http://api_inference:81/do_inference/penguins/", json=data)

        if response.ok:
            st.write(response.content)
        else:
            st.write("failed to do inference")
