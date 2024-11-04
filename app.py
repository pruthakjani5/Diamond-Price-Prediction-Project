import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import clipboard
#from mapie.regression import MapieRegressor

# Load the trained model
model = joblib.load('xgb.pkl')  # Replace with your model file

# Define the Streamlit app
def diamond_price_predictor():
    st.set_page_config(
        page_title="Diamond Price Predictor",
        page_icon="💎",
        layout="wide"
    )

    st.title("Diamond Price Prediction")

    # Sidebar with input fields and additional information
    with st.sidebar:
        st.header("About the Project")
        st.markdown(
            """
            **Problem Statement:**

            This project aims to predict the price of cubic zirconia stones based on their attributes. The goal is to help Gem Stones Co. Ltd. identify higher-profit stones and optimize their pricing strategy.

            **Features:**

            - **Carat:** Weight of the diamond (in carats).
            - **Cut:** Quality of the diamond cut.
            - **Color:** Color grade of the diamond.
            - **Clarity:** Clarity grade of the diamond.
            - **Depth:** Depth of the diamond (in millimeters).
            - **Table:** Width of the diamond's table.
            - **X, Y, Z:** Dimensions of the diamond (in millimeters).

            **Model:**

            A hypertuned regression model was trained on the [dataset](https://www.kaggle.com/datasets/colearninglounge/gemstone-price-prediction) to predict the price.

            **Note:** This model is for educational purposes only and may not provide accurate predictions in all cases.
            """
        )

        carat = st.number_input("Carat", min_value=0.1, max_value=10.0, value=1.0)
        cut = st.selectbox("Cut", options=["Fair", "Good", "Very Good", "Premium", "Ideal"])
        color = st.selectbox("Color", options=["D", "E", "F", "G", "H", "I", "J"])
        clarity = st.selectbox("Clarity", options=["I1", "IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2"])
        depth = st.number_input("Depth", min_value=0.0, max_value=100.0, value=62.0)
        table = st.number_input("Table", min_value=0.0, max_value=100.0, value=57.0)
        x = st.number_input("X dimension", min_value=0.0, max_value=100.0, value=5.0)
        y = st.number_input("Y dimension", min_value=0.0, max_value=100.0, value=5.0)
        z = st.number_input("Z dimension", min_value=0.0, max_value=100.0, value=3.0)

    # Main content area
    st.header("by Pruthak Jani")

    # Create a DataFrame with user input
    data = pd.DataFrame({
        'carat': [carat],
        'cut': [cut],
        'color': [color],
        'clarity': [clarity],
        'depth': [depth],
        'table': [table],
        'x': [x],
        'y': [y],
        'z': [z]
    })

    # Apply feature processing
    categorical_cols = ['cut', 'color', 'clarity']
    numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

    # Define custom ranking for ordinal variables
    cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
    clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

    # Numerical Pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
        # ('scaler', StandardScaler())
    ])

    # Categorical Pipeline
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinalencoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories]))
        # ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('num_pipeline', num_pipeline, numerical_cols),
        ('cat_pipeline', cat_pipeline, categorical_cols)
    ])

    # Transform the data
    processed_data = preprocessor.fit_transform(data)

    # Make prediction
    predicted_price = model.predict(processed_data)[0]

    # Display the predicted price
    st.markdown(f"## Predicted Diamond Price: ${predicted_price:.2f}")    

    # Add information about categories
    st.subheader("Category Details")
    st.markdown(
        """
        **Cut:**

        - **Fair:** Poorly cut diamonds with minimal brilliance and fire.
        - **Good:** Diamonds with good proportions but may have some minor flaws.
        - **Very Good:** Diamonds with excellent proportions and good brilliance.
        - **Premium:** Diamonds with excellent proportions and very good brilliance.
        - **Ideal:** Diamonds with the best possible proportions and maximum brilliance.

        **Color:**

        The GIA color grading system ranges from D (colorless) to Z (light yellow). Higher letters represent a more noticeable yellow tint.

        **Clarity:**

        The GIA clarity grading system ranges from Flawless (no inclusions) to Included (visible inclusions). Higher clarity grades generally indicate fewer imperfections.

        **Carat:**

        The carat is a unit of weight used to measure gemstones, including diamonds. One carat is equal to 200 milligrams.

        **Depth:**

        The depth of a diamond is its height in millimeters, measured from the culet (bottom) to the table (top). The depth-to-width ratio (depth divided by girdle diameter) can affect a diamond's brilliance.

        **Table:**

        The table is the flat facet at the top of a diamond. It affects the diamond's brilliance and fire.

        **X, Y, Z:**

        These dimensions represent the length, width, and depth of the diamond, respectively.
        """
    )

    # Add a button to copy the result to clipboard
    st.button("Copy Result to Clipboard", on_click=lambda: clipboard.copy(f"Predicted Price: ${predicted_price:.2f}"))

if __name__ == '__main__':
    st.session_state.copy_result = None
    diamond_price_predictor()
