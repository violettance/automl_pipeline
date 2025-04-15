import streamlit as st
import pandas as pd
import os
import time
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_categorical_dtype
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, save_model
import pygwalker as pyg
import streamlit.components.v1 as components

# Sayfa düzenini geniş olarak ayarla
st.set_page_config(layout="wide")

# ---------------- SESSION STATE INIT ----------------
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

if 'model_path' not in st.session_state:
    st.session_state.model_path = None

if 'data_path' not in st.session_state:
    st.session_state.data_path = None

# To store the profile report as HTML:
if 'profile_html' not in st.session_state:
    st.session_state.profile_html = None
if 'profile_generated' not in st.session_state:
    st.session_state.profile_generated = False

# To prevent reading the same file repeatedly:
if 'last_uploaded_file_name' not in st.session_state:
    st.session_state.last_uploaded_file_name = None

# Variable to hold the current DataFrame:
df = None

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.image("logo.png")
    st.title("AutoMl Tool")
    choice = st.radio("Navigation", ["Generate Report", "Visualization", "Model Builder"])
    st.info("""
### This interactive web app helps you automate your machine learning workflow — no coding required.

Just upload a CSV file to:
- **Explore your data** with Pandas Profiling  
- **Visualize it** using Pygwalker  
- **Train classification models** via PyCaret  

---

**🤖 Tech Stack:**
- Streamlit: Web UI  
- Pandas Profiling: Exploratory Data Analysis  
- PyCaret: AutoML for classification  
- Pygwalker: Interactive visual exploration  
""")

# ---------------- LOAD EXISTING CSV IF ALREADY IN session_state ----------------
def load_existing_csv():
    """
    Reads CSV from st.session_state.data_path, returns a DataFrame or None.
    """
    if st.session_state.data_path and os.path.exists(st.session_state.data_path):
        try:
            return pd.read_csv(st.session_state.data_path)
        except:
            return None
    return None

if choice != "Generate Report":
    # If we are not on the Generate Report tab,
    # let's load the CSV from session_state (if it exists).
    existing_df = load_existing_csv()
    if existing_df is not None:
        df = existing_df

# ---------------- GENERATE REPORT SECTION ----------------
if choice == "Generate Report":
    st.subheader("Exploratory Data Analysis with Pandas Profiling")
    st.markdown("Upload your CSV dataset below. Once uploaded, you can generate a comprehensive EDA report.")

    file = st.file_uploader("Upload Your Dataset Here", type=['csv'])

    if file:
        # If the file is new, only then will we read it
        if file.name != st.session_state.last_uploaded_file_name:
            # 1. Read file
            try:
                file.seek(0)
                df_new = pd.read_csv(file)
            except Exception as e:
                st.error(f"❌ Failed to read the uploaded file: {e}")
                df_new = None

            # 2. Save file
            if df_new is not None:
                timestamp = int(time.time())
                filename = f"sourcedata_{timestamp}.csv"
                try:
                    df_new.to_csv(filename, index=False)
                    st.session_state.data_path = filename
                    st.session_state.last_uploaded_file_name = file.name
                    df = df_new
                    st.success(f"✅ File uploaded and saved as: {filename}")
                    st.write(df)
                except Exception as e:
                    # Don't show the error message
                    # st.error(f"❌ Failed to save the uploaded file: {e}")
                    # İsteğe bağlı olarak hatayı loglayabilir veya sessizce geçebilirsiniz.
                    pass # Hata durumunda hiçbir şey yapma
        else:
            # If the same file was uploaded again
            if df is None or df.empty:
                # Try to load from session_state if available
                existing_df = load_existing_csv()
                if existing_df is not None:
                    df = existing_df
                else:
                    st.info("You re-uploaded the same file, but it seems we can't load it. Try a different file?")
            else:
                st.info("You've already uploaded this file. Below is the same data:")
                st.write(df)

    else:
        # file = None; show data if it exists in session_state
        if df is not None and not df.empty:
            st.write(df)
        else:
            st.info("Please upload a CSV file.")

    # Profile report generation button
    if st.button("Generate Profile Report", disabled=(df is None or df.empty)):
        try:
            total_duration = 10  # simulate 9 seconds total
            step1_duration = total_duration * 0.3
            step2_duration = total_duration * 0.3
            step3_duration = total_duration * 0.4

            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1
            status_text.info("🔍 Step 1: Initializing report generation...")
            for i in range(0, 31, 3):
                time.sleep(step1_duration / 10)
                progress_bar.progress(i)

            # Step 2
            status_text.info("🌀 Step 2: Analyzing variable distributions...")
            for i in range(31, 61, 3):
                time.sleep(step2_duration / 10)
                progress_bar.progress(i)

            # Step 3
            status_text.info("📈 Step 3: Building profiling report...")
            for i in range(61, 100, 4):
                time.sleep(step3_duration / 10)
                progress_bar.progress(i)

            profile_report = ProfileReport(df, title="Pandas Profiling Report")
            html_data = profile_report.to_html()
            st.session_state.profile_html = html_data
            st.session_state.profile_generated = True
            progress_bar.progress(100)
            status_text.success("✅ Report successfully generated!")
            components.html(html_data, height=800, scrolling=True)
        except Exception as e:
            st.error(f"Error generating profile report: {e}")

    # Show the report if it was previously generated
    if st.session_state.profile_generated and st.session_state.profile_html:
        st.markdown("#### Existing Profile Report")
        components.html(st.session_state.profile_html, height=800, scrolling=True)

# ---------------- VISUALIZATION SECTION ----------------
if choice == "Visualization":
    if df is not None and not df.empty:
        try:
            pyg_html = pyg.walk(df, return_html=True)
            components.html(pyg_html, height=1000, scrolling=True)
        except Exception as e:
            st.error(f"Error generating visualization: {e}")
    else:
        st.warning("Please upload a dataset first in the 'Generate Report' section.")

# ---------------- ML SECTION / MODEL BUILDER SECTION ----------------
if choice == "Model Builder":
    st.subheader("Automated Machine Learning with PyCaret")
    st.markdown("""
    In this section:
    - Select your **target column** from the dropdown below.
    - Click the **Train Model** button to automatically run and compare multiple classification algorithms.
    - The best-performing model will be selected and saved for you.
    - You will also be able to download this trained model to use later for prediction tasks.

    ⚙️ The following classification models are automatically evaluated:
    Logistic Regression, Decision Tree, Random Forest, Extra Trees, K Neighbors, Ridge Classifier, 
    AdaBoost, Gradient Boosting, SVM, Naive Bayes, Linear Discriminant Analysis, 
    LightGBM, Dummy Classifier, and Quadratic Discriminant Analysis.

    📌 Note: Make sure your target column has at least two records for each class.
    """)

    if df is not None and not df.empty:
        target = st.selectbox("Select the target variable", df.columns.tolist())

        # TRAIN MODEL (always active)
        if st.button("Train Models"):
            with st.spinner("Training models... Please wait."):
                try:
                    setup(data=df, target=target, session_id=123)
                    best_model = compare_models()

                    model_filename = f"model_{target}_{int(time.time())}"
                    model_path = f"{model_filename}.pkl"
                    save_model(best_model, model_filename)

                    st.session_state.model_trained = True
                    st.session_state.model_path = model_path

                    st.success(f"✅ Model trained and saved as: {model_path}")
                    st.balloons()
                except Exception as e:
                    st.error("🚨 The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2. Please try another target variable.")
                

        # DOWNLOAD button:
        #   - Active if model_path exists and the file exists on disk
        #   - Otherwise inactive
        if st.session_state.model_path:
            if os.path.exists(st.session_state.model_path):
                with open(st.session_state.model_path, "rb") as f:
                    st.download_button(
                        label="Download Best Model",
                        data=f,
                        file_name=st.session_state.model_path,
                        mime="application/octet-stream",
                        disabled=False
                    )
            else:
                st.warning("Model file not found on disk. Train again or check file path.")
        else:
            st.download_button(
                label="Download Best Model",
                data=b"",
                file_name="no_model.pkl",
                disabled=True
            )
    else:
        st.warning("Please upload a dataset first in the 'Generate Report' section.")