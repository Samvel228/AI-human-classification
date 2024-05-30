import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Загрузка модели
model = load_model('./model.keras')

# Функция для предсказания класса изображения
def predict_image(img):
    img = img.resize((224, 224))  # Убедитесь, что размер изображения соответствует входному размеру модели
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Добавьте ось для партии
    predictions = model.predict(img_array)
    return predictions

# Интерфейс Streamlit
st.title("Image Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    predictions = predict_image(image)
    class_names = ['AI', 'Human']  # Замените на имена ваших классов
    predicted_class = class_names[np.argmax(predictions)]
    st.write(f"Predicted class: {predicted_class}")
    st.write(f"Prediction probabilities: {predictions}")

