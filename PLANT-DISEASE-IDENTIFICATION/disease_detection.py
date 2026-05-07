import streamlit as st
import tensorflow as tf
import numpy as np
from disease_info import disease_data
from PIL import Image
import os

# PDF IMPORTS
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet


# =========================
# BASE DIRECTORY
# =========================
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# =========================
# LOAD MODEL
# =========================
model_path = os.path.join(
    BASE_DIR,
    "trained_plant_disease_model.keras"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

# =========================
# CLASS NAMES
# =========================
class_name = [

    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',

    'Blueberry___healthy',

    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',

    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',

    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',

    'Orange___Haunglongbing_(Citrus_greening)',

    'Peach___Bacterial_spot',
    'Peach___healthy',

    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',

    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',

    'Raspberry___healthy',

    'Soybean___healthy',

    'Squash___Powdery_mildew',

    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',

    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# =========================
# PREDICTION FUNCTION
# =========================
def model_prediction(test_image):

    image = tf.keras.preprocessing.image.load_img(
        test_image,
        target_size=(128,128)
    )

    input_arr = tf.keras.preprocessing.image.img_to_array(
        image
    )

    input_arr = np.array([input_arr])

    predictions = model.predict(input_arr)

    result_index = np.argmax(predictions)

    return result_index


# =========================
# PDF GENERATION FUNCTION
# =========================
def generate_pdf(
    disease_name,
    info
):

    pdf_file = "Disease_Report.pdf"

    doc = SimpleDocTemplate(pdf_file)

    styles = getSampleStyleSheet()

    elements = []

    # TITLE
    elements.append(
        Paragraph(
            "Plant Disease Detection Report",
            styles['Title']
        )
    )

    elements.append(Spacer(1, 20))

    # DISEASE
    elements.append(
        Paragraph(
            f"<b>Disease:</b> {disease_name}",
            styles['BodyText']
        )
    )

    elements.append(Spacer(1, 15))

    # DESCRIPTION
    elements.append(
        Paragraph(
            f"<b>Description:</b> {info['description']}",
            styles['BodyText']
        )
    )

    elements.append(Spacer(1, 12))

    # SYMPTOMS
    symptoms = "<br/>".join(info["symptoms"])

    elements.append(
        Paragraph(
            f"<b>Symptoms:</b><br/>{symptoms}",
            styles['BodyText']
        )
    )

    elements.append(Spacer(1, 12))

    # CAUSES
    causes = "<br/>".join(info["causes"])

    elements.append(
        Paragraph(
            f"<b>Causes:</b><br/>{causes}",
            styles['BodyText']
        )
    )

    elements.append(Spacer(1, 12))

    # PREVENTION
    prevention = "<br/>".join(info["prevention"])

    elements.append(
        Paragraph(
            f"<b>Prevention:</b><br/>{prevention}",
            styles['BodyText']
        )
    )

    elements.append(Spacer(1, 12))

    # CURE
    cure = "<br/>".join(info["cure"])

    elements.append(
        Paragraph(
            f"<b>Treatment / Cure:</b><br/>{cure}",
            styles['BodyText']
        )
    )

    doc.build(elements)

    return pdf_file


# =========================
# MAIN PAGE FUNCTION
# =========================
def disease_detection_page():

    st.header("🌱 Plant Disease Recognition")

    image_path = os.path.join(
        BASE_DIR,
        "Diseases.png"
    )

    img = Image.open(image_path)

    st.image(img)

    # =========================
    # IMAGE UPLOAD
    # =========================
    test_image = st.file_uploader(
        "Upload a Leaf Image",
        type=["jpg", "jpeg", "png"]
    )

    # =========================
    # SHOW IMAGE
    # =========================
    if test_image is not None:

        st.image(
            test_image,
            caption="Uploaded Leaf Image",
            width=300
        )

    # =========================
    # PREDICT BUTTON
    # =========================
    if st.button("Predict"):

        if test_image is None:

            st.warning(
                "Please upload an image first."
            )

        else:

            st.write("### Analyzing image...")

            result_index = model_prediction(
                test_image
            )

            predicted_disease = class_name[
                result_index
            ]

            # CLEAN DISPLAY NAME
            display_name = predicted_disease.replace(
                "___",
                " : "
            ).replace(
                "_",
                " "
            )

            # =========================
            # RESULT
            # =========================
            st.success(
                f"Model predicts this as {display_name}"
            )

            # =========================
            # DISEASE INFO
            # =========================
            if predicted_disease in disease_data:

                info = disease_data[
                    predicted_disease
                ]

                with st.expander(
                    f"About {display_name}",
                    expanded=True
                ):

                    st.subheader("Description")
                    st.write(info["description"])

                    st.subheader("Symptoms")

                    for symptom in info["symptoms"]:
                        st.write(f"• {symptom}")

                    st.subheader("Causes")

                    for cause in info["causes"]:
                        st.write(f"• {cause}")

                    st.subheader("Prevention")

                    for prevention in info["prevention"]:
                        st.write(f"• {prevention}")

                    st.subheader("Treatment / Cure")

                    for cure in info["cure"]:
                        st.write(f"• {cure}")

                # =========================
                # PDF GENERATION
                # =========================
                pdf_file = generate_pdf(
                    display_name,
                    info
                )

                # =========================
                # DOWNLOAD BUTTON
                # =========================
                with open(pdf_file, "rb") as pdf:

                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf,
                        file_name="Disease_Report.pdf",
                        mime="application/pdf"
                    )

            else:

                st.error(
                    "Disease information not available."
                )