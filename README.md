# AutoML Tool

This is a Streamlit application that allows you to build an automated ML pipeline using Streamlit, Pandas Profiling, and PyCaret.

## Installation

1. Clone the repository.

2. Install the required packages by running the following command:

```bash
pip install -r requirements.txt
```
**Note:** It is recommended to create a virtual environment before installing the dependencies.

3. Run the application by executing the following command:

```
streamlit run app.py
```

4. The application will open in your default browser.

## Usage

- The sidebar contains the following navigation options:
- **Upload**: Allows you to upload your dataset for modeling.
- **Profile Report**: Generates an automated exploratory data analysis report using Pandas Profiling.
- **ML**: Performs machine learning tasks using PyCaret.
- **Visualization**: Visualizes the data using Pygwalker.
- **Download**: Downloads the trained ML model.

### Upload Your Data for Modeling:

- Click on the **Upload** option in the sidebar.
- Upload your dataset using the file uploader.
- The uploaded dataset will be displayed in a table format.

### Automated Exploratory Data Analysis:

- Select the **Profile Report** option in the sidebar.
- If a dataset has been uploaded, an automated exploratory data analysis report will be generated using Pandas Profiling.
- The report will be displayed in the Streamlit app.

### Data Visualization with Pygwalker:

- Select the **Visualization** option in the sidebar.
- If a dataset has been uploaded, the data will be visualized using Pygwalker.
- The visualization will be embedded in the Streamlit app.

### Machine Learning:

- Select the **ML** option in the sidebar.
- If a dataset has been uploaded, you can perform machine learning tasks using PyCaret.
- Select the target variable from the dropdown menu.
- Click the **Train Model** button to train the model.
- The trained model and its performance metrics will be displayed in the Streamlit app.
- The best model will be saved as `best_model.pkl`.

### Download ML:

- Select the **Download** option in the sidebar.
- If a dataset has been uploaded and a model has been trained, you can download the trained model.
- Click the **Download Model** button to download the `best_model.pkl` file.

**Note:** Make sure to replace `"logo.png"` with your own logo file in the code.

## Deployment

This project can be deployed using Streamlit, a powerful Python library for building interactive web applications. Follow the steps below to deploy the project on a server or cloud platform.

To find more information about deploying an app, click [here](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app).


## Contributing

Contributions are welcome! If you find any issues or have suggestions, please feel free to open an issue or submit a pull request.
