import requests
import streamlit as st
import pandas as pd
import time
import sys
import os

HOST=os.environ['HOST']

WILDERNESS = {"Cache": "Cache", "Commanche": "Commanche", "Rawah": "Rawah","Neota": "Neota",}
SOIL_TYPE= {"C2702":"C2702","C2703":"C2703","C2704":"C2704","C2705":"C2705","C2706":"C2706",
            "C2717":"C2717","C3501":"C3501","C3502":"C3502","C4201":"C4201","C4703":"C4703",
            "C4704":"C4704","C4744":"C4744","C4758":"C4758","C5101":"C5101","C5151":"C5151",
            "C6101":"C6101","C6102":"C6102","C6731":"C6731","C7101":"C7101","C7102":"C7102",
            "C7103":"C7103","C7201":"C7201","C7202":"C7202","C7700":"C7700","C7701":"C7701",
            "C7702":"C7702","C7709":"C7709","C7710":"C7710","C7745":"C7745","C7746":"C7746",
            "C7755":"C7755","C7756":"C7756","C7757":"C7757","C7790":"C7790","C8703":"C8703",
            "C8707":"C8707","C8708":"C8708","C8771":"C8771","C872":"C872", "C876":"C876"}


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
if st.button("Insert data in the database"):
        if file is not None:
            files = {"file": file.getvalue()}
            file_container = st.expander("Check your uploaded .csv")
            shows = pd.read_csv(file)
            file.seek(0)
            file_container.write(shows)
            response = requests.post("http://"+HOST+":8502/train_model/uploadfile/", files=files)
            st.success(response.json())
        else:
            st.warning("Please choose a file to upload.")

st.header("Section to Train Model")
if st.button("train model"):
        response = requests.get("http://"+HOST+":8502/train_model/train/")
        st.success(response.text)


st.header("Section to do Inference")
with st.form("my_form"):
    Elevation = st.text_input("Elevation")
    Aspect = st.text_input("Aspect")
    Slope = st.text_input("Slope")
    Horizontal_Distance_To_Hydrology = st.text_input("Horizontal Distance To Hydrology")
    Vertical_Distance_To_Hydrology = st.text_input("Vertical Distance To Hydrology")
    Horizontal_Distance_To_Roadways = st.text_input("Horizontal Distance To Roadways")
    Hillshade_9am = st.text_input("Hillshade 9am")
    Hillshade_Noon = st.text_input("Hillshade Noon")
    Hillshade_3pm = st.text_input("Hillshade 3pm")
    Horizontal_Distance_To_Fire_Points = st.text_input("Horizontal Distance To Fire Points")
    Wilderness_Area = st.selectbox("Choose the wilderness Area", [i for i in WILDERNESS.keys()])
    Soil_Type = st.selectbox("Choose the soil type", [i for i in SOIL_TYPE.keys()])

    submitted = st.form_submit_button("Submit")
    if submitted:
        data = {
            "Elevation": Elevation,
            "Aspect": Aspect,
            "Slope": Slope,
            "Horizontal_Distance_To_Hydrology": Horizontal_Distance_To_Hydrology,
            "Vertical_Distance_To_Hydrology": Vertical_Distance_To_Hydrology,
            "Horizontal_Distance_To_Roadways": Horizontal_Distance_To_Roadways,
            "Hillshade_9am": Hillshade_9am,
            "Hillshade_Noon": Hillshade_Noon,
            "Hillshade_3pm": Hillshade_3pm,
            "Horizontal_Distance_To_Fire_Points": Horizontal_Distance_To_Fire_Points,
            "Wilderness_Area": Wilderness_Area,
            "Soil_Type": Soil_Type  
        }

        response = requests.post("http://"+HOST+":8503/do_inference/covertype/", json=data)

        if response.ok:
            st.success(response.text)
        else:
            st.error("failed to do inference")

st.header("Section to Save Info")
if st.button("Save Info"):
        response = requests.get("http://"+HOST+":8504/store/convert_inference_csv/")

        if response.ok:
                st.success(response.text)
        else:
                st.error("failed to do inference")
                st.success(response.text)
if st.button("train with new data from csv"):
        response = requests.get("http://"+HOST+":8502/train_model/train_from_csv/")

        if response.ok:
                st.success(response.text)
        else:
                st.error("failed to do inference")
                st.success(response.text)