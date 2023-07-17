import streamlit as st
import pandas as pd
import os

# Import profiling capabilities
import ydata_profiling
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Import ML capabilities
from pycaret.classification import setup, compare_models, pull, save_model


# Import data visualization capabilities
import pygwalker as pyg
import streamlit.components.v1 as components

with st.sidebar:
    st.image("logo.png")
    st.title("AutoMl Tool")
    choice = st.radio("Navigation", ["Upload", "Profile Report", "ML","Visualization", "Download"])
    st.info("This application allows you to build an automated ML pipeline using Streamlit, Pandas Profiling, and PyCaret")

df = None  # Initialize the df variable

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.title("Upload Your Data for Modelling")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df = pd.read_csv(file)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

if choice == "Profile Report":
    if df is not None:  # Check if df is defined
        st.title("Automated Exploratory Data Analysis")
        profile_report = ProfileReport(df)
        st_profile_report(profile_report)
    else:
        st.warning("Please upload a dataset first to perform profiling.")


if  choice =="Visualization":
    if df is not None:
        st.title("Data Visualization with Pygwalker")
        pyg_html = pyg.walk(df, return_html=True) # Generate the HTML using Pygwalker
        components.html(pyg_html, width=1000, height=1000, scrolling=True)# Embed the generated HTML into the Streamlit app
    else:
        st.warning("Please upload a dataset first to perform visualization.")

if choice == "ML":
    if df is not None and not df.empty:  # Check if df is defined and not empty
        st.title("Machine Learning")
        target = st.selectbox("Select Your Target", df.columns)
        if st.button("Train Model"):
            setup(df, target=target)
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.info("This is the ML Model")
            st.dataframe(compare_df)
            save_model(best_model, 'best_model.pkl')
    else:
        st.warning("Please upload a dataset first to perform ML.")


if choice == "Download":
    if df is not None and not df.empty:  # Check if df is defined and not empty
        st.title("Download ML")
        if os.path.exists('best_model.pkl'):
            with open('best_model.pkl', 'rb') as f: 
                st.download_button('Download Model', f, file_name="best_model.pkl")
        else:
            st.warning("Please train a model first.")
    else:
        st.warning("Please upload a dataset first to perform the download.")


        