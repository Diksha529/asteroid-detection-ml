import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Asteroid Detection Using ML",
    page_icon="☄️",
    layout="wide"
)


# ============================================================
# 2. TITLE
# ============================================================

st.title("☄️ Asteroid Detection Using Machine Learning")

st.write(
    "This application predicts whether an asteroid is "
    "**Hazardous or Not Hazardous** using Machine Learning "
    "and NASA Live Asteroid Data."
)

st.divider()


# ============================================================
# 3. LOAD DATASET
# ============================================================

DATA_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\PROJECT\nasa_cleaned (1).csv"

try:
    df = pd.read_csv(DATA_PATH)

except Exception as e:
    st.error("Dataset could not be loaded.")
    st.write("Check that this file exists:")
    st.code(DATA_PATH)
    st.stop()


# ============================================================
# 4. DATASET INFORMATION
# ============================================================

st.header("📊 Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Rows", df.shape[0])

with col2:
    st.metric("Total Columns", df.shape[1])

with col3:
    st.metric(
        "Hazardous Column",
        "Yes" if "Hazardous" in df.columns else "No"
    )


with st.expander("View Dataset"):
    st.dataframe(
        df.head(20),
        width="stretch"
    )


# ============================================================
# 5. CHECK HAZARDOUS COLUMN
# ============================================================

if "Hazardous" not in df.columns:

    st.error(
        "The 'Hazardous' column was not found in your dataset."
    )

    st.stop()


# ============================================================
# 6. CREATE X AND Y
# ============================================================

X = df.select_dtypes(
    include=["number"]
).copy()

y = df["Hazardous"]


# Remove target from X
if "Hazardous" in X.columns:

    X = X.drop(
        columns=["Hazardous"]
    )


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median(numeric_only=True)
)


# ============================================================
# 8. CONVERT TARGET TO 0/1
# ============================================================

if y.dtype == "object":

    y = (
        y.astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {

        "true": 1,
        "false": 0,

        "yes": 1,
        "no": 0,

        "hazardous": 1,
        "not hazardous": 0,

        "1": 1,
        "0": 0
    }

    y = y.map(mapping)


# Remove invalid rows
valid_rows = y.notna()

X = X.loc[valid_rows]

y = y.loc[valid_rows]

y = y.astype(int)


# ============================================================
# 9. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)


# ============================================================
# 10. RANDOM FOREST
# ============================================================

rf_model = RandomForestClassifier(

    n_estimators=100,

    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(
    X_test
)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)


# ============================================================
# 11. DECISION TREE
# ============================================================

dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(
    X_train,
    y_train
)

dt_pred = dt_model.predict(
    X_test
)

dt_accuracy = accuracy_score(
    y_test,
    dt_pred
)


# ============================================================
# 12. LOGISTIC REGRESSION
# ============================================================

lr_model = LogisticRegression(
    max_iter=1000
)

lr_model.fit(
    X_train,
    y_train
)

lr_pred = lr_model.predict(
    X_test
)

lr_accuracy = accuracy_score(
    y_test,
    lr_pred
)


# ============================================================
# 13. MODEL ACCURACY
# ============================================================

st.header("🤖 Model Accuracy")

acc1, acc2, acc3 = st.columns(3)

with acc1:

    st.metric(
        "Random Forest",
        f"{rf_accuracy * 100:.2f}%"
    )

with acc2:

    st.metric(
        "Decision Tree",
        f"{dt_accuracy * 100:.2f}%"
    )

with acc3:

    st.metric(
        "Logistic Regression",
        f"{lr_accuracy * 100:.2f}%"
    )


# ============================================================
# 14. MODEL ACCURACY GRAPH
# ============================================================

st.subheader("📈 Model Accuracy Comparison")

models = [

    "Random Forest",

    "Decision Tree",

    "Logistic Regression"
]

accuracies = [

    rf_accuracy * 100,

    dt_accuracy * 100,

    lr_accuracy * 100
]

fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.bar(
    models,
    accuracies
)

ax.set_xlabel(
    "ML Models"
)

ax.set_ylabel(
    "Accuracy (%)"
)

ax.set_title(
    "Asteroid Detection Model Comparison"
)

ax.set_ylim(
    0,
    100
)

plt.xticks(
    rotation=15
)

st.pyplot(fig)


# ============================================================
# 15. SELECT BEST MODEL
# ============================================================

accuracy_values = {

    "Random Forest":
        rf_accuracy,

    "Decision Tree":
        dt_accuracy,

    "Logistic Regression":
        lr_accuracy
}

best_model_name = max(
    accuracy_values,
    key=accuracy_values.get
)

st.success(

    f"🏆 Best Model: {best_model_name} "
    f"with "
    f"{accuracy_values[best_model_name] * 100:.2f}% accuracy"
)


# ============================================================
# 16. NEW ASTEROID PREDICTION
# ============================================================

st.divider()

st.header(
    "☄️ New Asteroid Prediction"
)

st.write(
    "Enter the asteroid's numerical values below "
    "to predict whether it is Hazardous."
)


input_values = {}

input_columns = st.columns(2)

for i, feature in enumerate(X.columns):

    with input_columns[i % 2]:

        default_value = float(
            X[feature].median()
        )

        input_values[feature] = st.number_input(

            feature,

            value=default_value,

            format="%.6f"
        )


# ============================================================
# 17. PREDICT NEW ASTEROID
# ============================================================

if st.button(
    "🔍 Predict Asteroid",
    type="primary"
):

    new_asteroid = pd.DataFrame(

        [input_values],

        columns=X.columns
    )

    prediction = rf_model.predict(
        new_asteroid
    )

    probability = rf_model.predict_proba(
        new_asteroid
    )

    hazardous_probability = (
        probability[0][1] * 100
    )

    safe_probability = (
        probability[0][0] * 100
    )

    st.divider()

    st.subheader(
        "🎯 Prediction Result"
    )

    if prediction[0] == 1:

        st.error(
            "🔴 ASTEROID IS HAZARDOUS"
        )

    else:

        st.success(
            "🟢 ASTEROID IS NOT HAZARDOUS"
        )

    p1, p2 = st.columns(2)

    with p1:

        st.metric(
            "Not Hazardous Probability",
            f"{safe_probability:.2f}%"
        )

    with p2:

        st.metric(
            "Hazardous Probability",
            f"{hazardous_probability:.2f}%"
        )

    # --------------------------------------------------------
    # Probability Graph
    # --------------------------------------------------------

    st.subheader(
        "📊 Prediction Probability"
    )

    probability_df = pd.DataFrame({

        "Class": [

            "Not Hazardous",

            "Hazardous"
        ],

        "Probability": [

            safe_probability,

            hazardous_probability
        ]
    })

    fig2, ax2 = plt.subplots(
        figsize=(8, 5)
    )

    ax2.bar(

        probability_df["Class"],

        probability_df["Probability"]
    )

    ax2.set_ylabel(
        "Probability (%)"
    )

    ax2.set_title(
        "Asteroid Hazard Prediction Probability"
    )

    ax2.set_ylim(
        0,
        100
    )

    st.pyplot(fig2)


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

st.divider()

st.header(
    "🔲 Random Forest Confusion Matrix"
)

cm = confusion_matrix(
    y_test,
    rf_pred
)

fig3, ax3 = plt.subplots()

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(
    ax=ax3
)

ax3.set_title(
    "Random Forest - Asteroid Hazard Classification"
)

st.pyplot(fig3)


# ============================================================
# 19. CLASSIFICATION REPORT
# ============================================================

st.subheader(
    "📋 Random Forest Classification Report"
)

report = classification_report(

    y_test,

    rf_pred,

    output_dict=True,

    zero_division=0
)

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df,
    width="stretch"
)


# ============================================================
# 20. FEATURE IMPORTANCE
# ============================================================

st.divider()

st.header(
    "⭐ Top Important Features"
)

importance = pd.Series(

    rf_model.feature_importances_,

    index=X.columns

).sort_values(
    ascending=False
)

top_features = importance.head(10)

fig4, ax4 = plt.subplots(
    figsize=(10, 5)
)

top_features.plot(
    kind="bar",
    ax=ax4
)

ax4.set_title(
    "Top 10 Important Features for Asteroid Detection"
)

ax4.set_xlabel(
    "Features"
)

ax4.set_ylabel(
    "Importance"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

st.pyplot(fig4)


# ============================================================
# 21. COMPLETE MODEL PERFORMANCE
# ============================================================

st.divider()

st.header(
    "📊 Complete Model Performance Comparison"
)

comparison = pd.DataFrame({

    "Model": [

        "Random Forest",

        "Decision Tree",

        "Logistic Regression"
    ],

    "Accuracy": [

        accuracy_score(
            y_test,
            rf_pred
        ),

        accuracy_score(
            y_test,
            dt_pred
        ),

        accuracy_score(
            y_test,
            lr_pred
        )
    ],

    "Precision": [

        precision_score(
            y_test,
            rf_pred,
            zero_division=0
        ),

        precision_score(
            y_test,
            dt_pred,
            zero_division=0
        ),

        precision_score(
            y_test,
            lr_pred,
            zero_division=0
        )
    ],

    "Recall": [

        recall_score(
            y_test,
            rf_pred,
            zero_division=0
        ),

        recall_score(
            y_test,
            dt_pred,
            zero_division=0
        ),

        recall_score(
            y_test,
            lr_pred,
            zero_division=0
        )
    ],

    "F1 Score": [

        f1_score(
            y_test,
            rf_pred,
            zero_division=0
        ),

        f1_score(
            y_test,
            dt_pred,
            zero_division=0
        ),

        f1_score(
            y_test,
            lr_pred,
            zero_division=0
        )
    ]
})


comparison_percentage = comparison.copy()

comparison_percentage[

    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

] = comparison_percentage[

    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

] * 100


st.dataframe(

    comparison_percentage.style.format({

        "Accuracy": "{:.2f}%",

        "Precision": "{:.2f}%",

        "Recall": "{:.2f}%",

        "F1 Score": "{:.2f}%"
    }),

    width="stretch"
)


# ============================================================
# 22. COMPLETE PERFORMANCE GRAPH
# ============================================================

fig5, ax5 = plt.subplots(
    figsize=(10, 6)
)

comparison.set_index(
    "Model"
)[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
].plot(
    kind="bar",
    ax=ax5
)

ax5.set_title(
    "Asteroid Detection - Model Performance Comparison"
)

ax5.set_ylabel(
    "Score"
)

ax5.set_xlabel(
    "ML Model"
)

ax5.set_ylim(
    0,
    1
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

st.pyplot(fig5)


# ============================================================
# 23. NASA LIVE ASTEROID API
# ============================================================

st.divider()

st.header(
    "🚀 NASA Live Asteroid Data"
)

st.write(
    "Fetch real-time Near-Earth Asteroid data "
    "from NASA NeoWs API."
)


# ============================================================
# 24. NASA API INPUT
# ============================================================

api_key = st.text_input(

    "NASA API Key",

    value="DEMO_KEY",

    type="password"
)

selected_date = st.date_input(
    "Select Date"
)


# ============================================================
# 25. FETCH NASA DATA
# ============================================================

if st.button(
    "🌎 Fetch NASA Live Data"
):

    date_string = selected_date.strftime(
        "%Y-%m-%d"
    )

    nasa_url = (
        "https://api.nasa.gov/neo/rest/v1/feed"
    )

    params = {

        "start_date":
            date_string,

        "end_date":
            date_string,

        "api_key":
            api_key
    }

    try:

        response = requests.get(

            nasa_url,

            params=params,

            timeout=30
        )

        # ----------------------------------------------------
        # API STATUS
        # ----------------------------------------------------

        if response.status_code != 200:

            st.error(
                f"NASA API Error: "
                f"{response.status_code}"
            )

            st.stop()


        # ----------------------------------------------------
        # GET JSON DATA
        # ----------------------------------------------------

        nasa_data = response.json()

        asteroids = (

            nasa_data

            .get(
                "near_earth_objects",
                {}
            )

            .get(
                date_string,
                []
            )
        )


        if len(asteroids) == 0:

            st.warning(
                "No asteroid data found for this date."
            )

            st.stop()


        st.success(

            f"NASA API Connected Successfully! "
            f"{len(asteroids)} asteroids found."
        )


        # ====================================================
        # 26. CREATE LIVE ASTEROID TABLE
        # ====================================================

        live_rows = []


        for asteroid in asteroids:

            diameter_data = (

                asteroid

                .get(
                    "estimated_diameter",
                    {}
                )

                .get(
                    "kilometers",
                    {}
                )
            )


            approach_data = (

                asteroid

                .get(
                    "close_approach_data",
                    []
                )
            )


            if len(approach_data) > 0:

                approach = approach_data[0]


                velocity_km_s = float(

                    approach

                    .get(
                        "relative_velocity",
                        {}
                    )

                    .get(
                        "kilometers_per_second",
                        0
                    )
                )


                miss_distance_km = float(

                    approach

                    .get(
                        "miss_distance",
                        {}
                    )

                    .get(
                        "kilometers",
                        0
                    )
                )

            else:

                velocity_km_s = 0

                miss_distance_km = 0


            live_rows.append({

                "ID":
                    asteroid.get(
                        "id",
                        ""
                    ),

                "Name":
                    asteroid.get(
                        "name",
                        ""
                    ),

                "NASA Hazardous":
                    asteroid.get(
                        "is_potentially_hazardous_asteroid",
                        False
                    ),

                "Absolute Magnitude":
                    asteroid.get(
                        "absolute_magnitude_h",
                        0
                    ),

                "Diameter Min KM":
                    diameter_data.get(
                        "estimated_diameter_min",
                        0
                    ),

                "Diameter Max KM":
                    diameter_data.get(
                        "estimated_diameter_max",
                        0
                    ),

                "Velocity KM/S":
                    velocity_km_s,

                "Miss Distance KM":
                    miss_distance_km
            })


        live_df = pd.DataFrame(
            live_rows
        )


        # ====================================================
        # 27. DISPLAY LIVE DATA
        # ====================================================

        st.subheader(
            "🌌 NASA Live Asteroid Table"
        )

        st.dataframe(
            live_df,
            width="stretch"
        )


        # ====================================================
        # 28. LIVE ASTEROID GRAPH
        # ====================================================

        st.subheader(
            "📈 NASA Live Asteroid Data Graph"
        )

        graph_df = live_df.head(10)

        fig6, ax6 = plt.subplots(
            figsize=(10, 5)
        )

        ax6.bar(

            graph_df["Name"].astype(str),

            graph_df["Velocity KM/S"]
        )

        ax6.set_xlabel(
            "Asteroid"
        )

        ax6.set_ylabel(
            "Velocity (KM/S)"
        )

        ax6.set_title(
            "NASA Live Asteroid Velocity"
        )

        plt.xticks(
            rotation=45
        )

        plt.tight_layout()

        st.pyplot(fig6)


        # ====================================================
        # 29. SELECT LIVE ASTEROID
        # ====================================================

        asteroid_names = (
            live_df["Name"].tolist()
        )

        selected_asteroid = st.selectbox(

            "☄️ Select an asteroid "
            "for ML prediction",

            asteroid_names
        )


        selected_row = live_df[
            live_df["Name"] ==
            selected_asteroid
        ].iloc[0]


        # ====================================================
        # 30. SELECTED ASTEROID DETAILS
        # ====================================================

        st.subheader(
            f"🛰️ Selected Asteroid: "
            f"{selected_asteroid}"
        )


        c1, c2, c3, c4 = st.columns(4)


        with c1:

            st.metric(

                "NASA Hazard Flag",

                str(
                    selected_row[
                        "NASA Hazardous"
                    ]
                )
            )


        with c2:

            st.metric(

                "Diameter Max KM",

                f"{selected_row['Diameter Max KM']:.6f}"
            )


        with c3:

            st.metric(

                "Velocity KM/S",

                f"{selected_row['Velocity KM/S']:.6f}"
            )


        with c4:

            st.metric(

                "Miss Distance KM",

                f"{selected_row['Miss Distance KM']:.2f}"
            )


        # ====================================================
        # 31. NASA DATA → ML MODEL
        # ====================================================

        st.subheader(
            "🤖 ML Prediction on NASA Live Asteroid"
        )

        st.write(
            "NASA se aaya hua asteroid data "
            "Random Forest model ko diya ja raha hai."
        )


        nasa_input = {}


        for feature in X.columns:

            feature_lower = (
                feature.lower()
            )


            # ----------------------------------------------
            # Absolute Magnitude
            # ----------------------------------------------

            if (
                "absolute" in feature_lower
                and "magnitude" in feature_lower
            ):

                value = selected_row[
                    "Absolute Magnitude"
                ]


            # ----------------------------------------------
            # Diameter Minimum
            # ----------------------------------------------

            elif (
                "diameter" in feature_lower
                and "min" in feature_lower
            ):

                value = selected_row[
                    "Diameter Min KM"
                ]


            # ----------------------------------------------
            # Diameter Maximum
            # ----------------------------------------------

            elif (
                "diameter" in feature_lower
                and "max" in feature_lower
            ):

                value = selected_row[
                    "Diameter Max KM"
                ]


            # ----------------------------------------------
            # Velocity
            # ----------------------------------------------

            elif (
                "velocity" in feature_lower
                or "relative_velocity" in feature_lower
            ):

                value = selected_row[
                    "Velocity KM/S"
                ]


            # ----------------------------------------------
            # Miss Distance
            # ----------------------------------------------

            elif (
                "miss" in feature_lower
                or "distance" in feature_lower
            ):

                value = selected_row[
                    "Miss Distance KM"
                ]


            # ----------------------------------------------
            # Unknown Feature
            # ----------------------------------------------

            else:

                value = float(
                    X[feature].median()
                )


            nasa_input[feature] = value


        nasa_input_df = pd.DataFrame(

            [nasa_input],

            columns=X.columns
        )


        # ====================================================
        # 32. SHOW VALUES SENT TO MODEL
        # ====================================================

        with st.expander(
            "View data sent to ML model"
        ):

            st.dataframe(
                nasa_input_df,
                width="stretch"
            )


        # ====================================================
        # 33. NASA ASTEROID ML PREDICTION
        # ====================================================

        nasa_prediction = rf_model.predict(

            nasa_input_df
        )


        nasa_probability = (

            rf_model.predict_proba(
                nasa_input_df
            )
        )


        nasa_hazard_probability = (

            nasa_probability[0][1] * 100
        )


        nasa_safe_probability = (

            nasa_probability[0][0] * 100
        )


        st.subheader(
            "🎯 NASA Live Asteroid ML Result"
        )


        if nasa_prediction[0] == 1:

            st.error(

                "🔴 ML MODEL PREDICTION: "
                "ASTEROID IS HAZARDOUS"
            )

        else:

            st.success(

                "🟢 ML MODEL PREDICTION: "
                "ASTEROID IS NOT HAZARDOUS"
            )


        # ====================================================
        # 34. NASA ML PROBABILITY
        # ====================================================

        n1, n2 = st.columns(2)


        with n1:

            st.metric(

                "Not Hazardous",

                f"{nasa_safe_probability:.2f}%"
            )


        with n2:

            st.metric(

                "Hazardous",

                f"{nasa_hazard_probability:.2f}%"
            )


        # ====================================================
        # 35. NASA ML PROBABILITY GRAPH
        # ====================================================

        st.subheader(
            "📊 NASA Asteroid ML Prediction Probability"
        )


        nasa_probability_df = pd.DataFrame({

            "Class": [

                "Not Hazardous",

                "Hazardous"
            ],

            "Probability": [

                nasa_safe_probability,

                nasa_hazard_probability
            ]
        })


        fig7, ax7 = plt.subplots(
            figsize=(8, 5)
        )


        ax7.bar(

            nasa_probability_df["Class"],

            nasa_probability_df["Probability"]
        )


        ax7.set_ylabel(
            "Probability (%)"
        )

        ax7.set_title(
            "NASA Live Asteroid ML Prediction"
        )

        ax7.set_ylim(
            0,
            100
        )


        st.pyplot(fig7)


        # ====================================================
        # 36. IMPORTANT NOTE
        # ====================================================

        st.info(

            "NASA Hazard Flag aur ML Model Prediction "
            "do alag results hain. NASA ka flag NASA ke "
            "official data se aata hai, jabki ML prediction "
            "tumhare trained Random Forest model se aata hai."
        )


    except requests.exceptions.RequestException as e:

        st.error(
            f"NASA API connection error: {e}"
        )


    except Exception as e:

        st.error(
            f"Unexpected error: {e}"
        )


# ============================================================
# 37. PROJECT WORKFLOW
# ============================================================

st.divider()

st.header(
    "🔄 Project Workflow"
)

st.write(
    """
    Dataset
    ↓
    Data Cleaning & Preprocessing
    ↓
    Data Visualization
    ↓
    Train-Test Split
    ↓
    Random Forest / Decision Tree / Logistic Regression
    ↓
    Accuracy Evaluation
    ↓
    Confusion Matrix
    ↓
    Classification Report
    ↓
    Feature Importance
    ↓
    New Asteroid Input
    ↓
    NASA Live API
    ↓
    Live Asteroid Selection
    ↓
    ML Prediction
    ↓
    Probability Graph
    ↓
    Final Result
    """
)


# ============================================================
# 38. FOOTER
# ============================================================

st.divider()

st.caption(
    "☄️ Asteroid Detection Using ML | "
    "Python + Machine Learning + Streamlit + NASA API"
)