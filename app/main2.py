# Import necessary libraries
import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import time
import urllib.parse


# Load pre-trained model and class indices
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)

# Load class indices (mapping of class index to class name)
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Dictionary containing causes and remedies for each plant disease
disease_info={
    "Apple___Apple_scab": {
        "cause": "Caused by the fungus Venturia inaequalis.",
        "remedy": "Use fungicides, prune and dispose of infected leaves, and maintain proper tree spacing.",
        "medicine": "Captan, Mancozeb, or Myclobutanil-based fungicides."
    },
    "Apple___Black_rot": {
        "cause": "Caused by the fungus Botryosphaeria obtusa.",
        "remedy": "Use fungicides, remove and destroy infected branches, and prune trees regularly.",
        "medicine": "Copper-based fungicides or Mancozeb."
    },
    "Apple___Cedar_apple_rust": {
        "cause": "Caused by the fungus Gymnosporangium juniperi-virginianae.",
        "remedy": "Use fungicides, remove nearby cedar/juniper hosts, and prune infected areas.",
        "medicine": "Myclobutanil or Mancozeb-based fungicides."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "cause": "Powdery mildew on cherry trees is typically caused by the fungus Podosphaera clandestina.",
        "remedy": "To manage powdery mildew, ensure good air circulation by properly pruning trees. Remove and destroy infected leaves, apply preventative fungicides, and maintain good orchard hygiene. Avoid overhead watering and ensure proper spacing between trees.",
        "medicine": "Bacillus subtilis."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "cause": "Caused by the fungus Cercospora zeina.",
        "remedy": "Use resistant hybrids, practice crop rotation, and apply fungicides if needed.",
        "medicine": "Strobilurins or Triazole-based fungicides."
    },
    "Corn_(maize)__Common_rust": {
        "cause": "Caused by the fungus Puccinia sorghi.",
        "remedy": "Use resistant hybrids, ensure proper field sanitation, and consider fungicides if the outbreak is severe.",
        "medicine": "Fungicides containing Azoxystrobin or Tebuconazole."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "cause": "Caused by the fungus Exserohilum turcicum.",
        "remedy": "Use resistant hybrids, manage crop residues, and apply fungicides as needed.",
        "medicine": "Fungicides containing Mancozeb or Strobilurins."
    },
    "Grape___Black_rot": {
        "cause": "Caused by the fungus Guignardia bidwellii.",
        "remedy": "Use fungicides, remove and destroy infected plant parts, and ensure proper vineyard sanitation.",
        "medicine": "Fungicides containing Myclobutanil or Captan."
    },
    "Grape___Esca_(Black_Measles)": {
        "cause": "Caused by a complex of fungal pathogens.",
        "remedy": "Prune and remove infected wood, and apply fungicides as needed.",
        "medicine": "Fungicides containing Trifloxystrobin or Boscalid."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "cause": "Caused by the fungus Isariopsis clavispora.",
        "remedy": "Use fungicides, ensure proper vineyard sanitation, and apply fungicides as needed.",
        "medicine": "Copper-based fungicides or Mancozeb."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "cause": "Caused by the bacterium Candidatus Liberibacter asiaticus.",
        "remedy": "Control psyllids, remove infected trees, and apply insecticides.",
        "medicine": "Insecticides containing Imidacloprid or Thiamethoxam."
    },
    "Peach___Bacterial_spot": {
        "cause": "Caused by the bacterium Xanthomonas arboricola pv. pruni.",
        "remedy": "Use resistant varieties, apply copper-based sprays, and ensure proper pruning and sanitation.",
        "medicine": "Copper-based sprays."
    },
    "Pepper,_bell___Bacterial_spot": {
        "cause": "Caused by the bacterium Xanthomonas campestris pv. vesicatoria.",
        "remedy": "Use resistant varieties, apply copper-based sprays, and maintain proper plant spacing.",
        "medicine": "Copper-based sprays."
    },
    "Potato___Early_blight": {
        "cause": "Caused by the fungus Alternaria solani.",
        "remedy": "Use resistant varieties, rotate crops, and apply fungicides as needed.",
        "medicine": "Fungicides containing Mancozeb or Chlorothalonil."
    },
    "Potato___Late_blight": {
        "cause": "Caused by the fungus Phytophthora infestans.",
        "remedy": "Use resistant varieties, rotate crops, and apply fungicides as needed.",
        "medicine": "Fungicides containing Chlorothalonil or Mancozeb."
    },
    "Squash___Powdery_mildew": {
        "cause": "Caused by the fungus Erysiphe cichoracearum or Podosphaera xanthii.",
        "remedy": "Use resistant varieties, ensure proper ventilation, and apply fungicides if needed.",
        "medicine": "Fungicides containing Myclobutanil or Trifloxystrobin."
    },
    "Strawberry___Leaf_scorch": {
        "cause": "Caused by the fungus Diplocarpon earliana.",
        "remedy": "Ensure proper spacing, remove and destroy infected leaves, and apply fungicides if needed.",
        "medicine": "Fungicides containing Captan or Myclobutanil."
    },
    "Tomato___Bacterial_spot": {
        "cause": "Caused by the bacterium Xanthomonas gardneri.",
        "remedy": "Use resistant varieties, maintain proper plant spacing, and apply copper-based sprays.",
        "medicine": "Copper-based sprays."
    },
    "Tomato___Early_blight": {
        "cause": "Caused by the fungus Alternaria solani.",
        "remedy": "Use resistant varieties, rotate crops, and apply fungicides as needed.",
        "medicine": "Fungicides containing Mancozeb or Chlorothalonil."
    },
    "Tomato___Late_blight": {
        "cause": "Caused by the fungus Phytophthora infestans.",
        "remedy": "Use resistant varieties, rotate crops, and apply fungicides as needed.",
        "medicine": "Fungicides containing Chlorothalonil or Mancozeb."
    },
    "Tomato___Leaf_Mold": {
        "cause": "Caused by the fungus Cladosporium fulvum.",
        "remedy": "Ensure proper ventilation, remove and destroy infected leaves, and apply fungicides.",
        "medicine": "Fungicides containing Copper or Mancozeb."
    },
    "Tomato___Septoria_leaf_spot": {
        "cause": "Caused by the fungus Septoria lycopersici.",
        "remedy": "Use resistant varieties, rotate crops, and apply fungicides as needed.",
        "medicine": "Fungicides containing Chlorothalonil or Mancozeb."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "cause": "Caused by the mite Tetranychus urticae.",
        "remedy": "Use miticides, maintain proper humidity, and apply insecticidal soap as needed.",
        "medicine": "Miticides containing Abamectin or Bifenthrin."
    },
    "Tomato___Target_Spot": {
        "cause": "Caused by the fungus Corynespora cassiicola.",
        "remedy": "Ensure proper ventilation, apply fungicides, and practice crop rotation.",
        "medicine": "Fungicides containing Copper or Mancozeb."
    },
    "Tomato___Tomato_mosaic_virus": {
        "cause": "Caused by the Tomato Mosaic Virus (ToMV).",
        "remedy": "Practice good hygiene, remove infected plants, and disinfect tools.",
        "medicine": "No specific medicine, practice good cultural practices."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "cause": "Caused by the Tomato Yellow Leaf Curl Virus (TYLCV).",
        "remedy": "Control whiteflies, use resistant varieties, and ensure proper plant spacing.",
        "medicine": "Insecticides containing Imidacloprid or Thiamethoxam."
    }
}

# Function to Load and Preprocess the Image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit App
st.title("Leaf-Image Based Plant Disease Recognition")

# Add a background image with blur
background_image_url = "https://images.pexels.com/photos/1454794/pexels-photo-1454794.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("{background_image_url}");
    background-size: cover;
    background-position: center;
    backdrop-filter: blur(10px);  /* Adjust the blur radius as needed */
}}
[data-testid="stSidebar"] {{
    background: none;  /* Ensure sidebar doesn't have background */
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Streamlit elements inside the white box
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.markdown('<div class="container">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    # Progress bar to simulate processing time
    progress_bar = col2.progress(0)
    for perc_completed in range(100):
        time.sleep(0.005)
        progress_bar.progress(perc_completed + 1)

    with col1:
        resized_img = image.resize((200, 200))
        st.image(resized_img, caption="Uploaded Image")

    with col2:
        if st.button("Predict"):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f"Prediction: {prediction}")

            # Check if there's cause and remedy information
            if prediction in disease_info:
                cause = disease_info[prediction]["cause"]
                remedy = disease_info[prediction]["remedy"]
                medicine_name = disease_info[prediction]["medicine"]
                st.write(f"**Cause:** {cause}")
                st.write(f"**Remedy:** {remedy}")
                st.write(f"**Recommended medicine:** {medicine_name}")

                #To show amazon link
                base_google_shopping_search_url = "https://www.google.com/search?tbm=shop&q="
                encoded_medicine_name = urllib.parse.quote(medicine_name)
                search_url = base_google_shopping_search_url + encoded_medicine_name

                base_google_search_url = "https://www.google.com/search?q="
                encoded_disease_name = urllib.parse.quote(prediction)
                search_url2 = base_google_search_url + encoded_disease_name
                st.write((f"🛍️[Search for {medicine_name} on web]({search_url})"))
                st.write((f"📖[Read more about {prediction} on web]({search_url2})"))
            else:
                st.text("The plant is healthy☘️")
