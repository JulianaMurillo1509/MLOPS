import requests
import streamlit as st
import pandas as pd
import time
import sys
import os

HOST=os.environ['HOST']

WILDERNESS = {"Cache": 0, "Commanche": 1, "Rawah": 2,"Neota": 3,}
SOIL_TYPE= {"C2702":0,"C2703":1,"C2704":3,"C2705":4,"C2706":5,
            "C2717":6,"C3501":7,"C3502":8,"C4201":9,"C4703":10,
            "C4704":11,"C4744":12,"C4758":13,"C5101":14,"C5151":15,
            "C6101":16,"C6102":17,"C6731":18,"C7101":19,"C7102":20,
            "C7103":21,"C7201":22,"C7202":23,"C7700":24,"C7701":25,
            "C7702":26,"C7709":27,"C7710":28,"C7745":29,"C7746":30,
            "C7755":31,"C7756":32,"C7757":33,"C7790":34,"C8703":35,
            "C8707":36,"C8708":37,"C8771":38,"C872":39, "C876":40}


# https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("Proyecto 3 - Cover type web app")

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
            with st.spinner('Uploading data to database...'):
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

    Elevation = st.text_input("Elevation",value=2991)
    Aspect = st.text_input("Aspect",value=119)
    Slope = st.text_input("Slope",value=7)
    Horizontal_Distance_To_Hydrology = st.text_input("Horizontal Distance To Hydrology",value=67)
    Vertical_Distance_To_Hydrology = st.text_input("Vertical Distance To Hydrology",value=11)
    Horizontal_Distance_To_Roadways = st.text_input("Horizontal Distance To Roadways",value=1015)
    Hillshade_9am = st.text_input("Hillshade 9am",value=233)
    Hillshade_Noon = st.text_input("Hillshade Noon",value=234)
    Hillshade_3pm = st.text_input("Hillshade 3pm",value=11)
    Horizontal_Distance_To_Fire_Points = st.text_input("Horizontal Distance To Fire Points", value=1570)
    Wilderness_Area = st.selectbox("Choose the wilderness Area", [i for i in WILDERNESS.keys()], index=0)
    Soil_Type = st.selectbox("Choose the soil type", [i for i in SOIL_TYPE.keys()], index=0)


    # Convertir los valores seleccionados en los números correspondientes
    wilderness_num = WILDERNESS[Wilderness_Area]
    soil_type_num = SOIL_TYPE[Soil_Type]

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
            "Wilderness_Area": wilderness_num,
            "Soil_Type": soil_type_num  
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