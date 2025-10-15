import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image


st.set_page_config(
    page_title="♻️ Smart Garbage Classifier",
    page_icon="♻️",
    layout="wide"
)

# Load Model
model = tf.keras.models.load_model('model/garbage_classifier.keras',compile=False)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

st.title("♻️ Garbage Classification App")
st.write("Upload an image to predict its category.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds)]
    confidence = np.max(preds)

    st.write(f"### 🏷 Predicted: {pred_class}")
    
    st.write(f"### 🔢 Confidence: {confidence:.2f}")


# Sidebar

st.sidebar.title("ℹ️ About the App")
st.sidebar.markdown("""
This app classifies **waste images** into 6 categories using a pre-trained **MobileNetV2** model.

**Categories:**
- 📦 Cardboard  
- 🥂 Glass  
- 🧲 Metal  
- 📄 Paper  
- 🧃 Plastic  
- 🗑️ Trash  

Upload an image to see the prediction result!
""")

st.sidebar.markdown("---")
st.sidebar.caption("Developed with ❤️ using Streamlit & TensorFlow")


