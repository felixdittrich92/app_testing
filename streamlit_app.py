import streamlit as st

import io

import cv2
from PIL import Image

import textract

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img

CLASS_IDXS = ["not good", "good"]

@st.cache(allow_output_mutation=True)
def load_models():
  model_eval = load_model('models/doc_model.model')
  model_eval._make_predict_function()
  model_eval.summary()
  model_auto = load_model('models/auto_model.model')
  model_auto._make_predict_function()
  model_auto.summary()
  return model_eval, model_auto

@st.cache
def __calculate_score(y_pred_class, y_pred_prob):
  if y_pred_class == 0:
    MAX = 0.5
    scaled_percentage = (y_pred_prob * MAX) / 100
    return MAX - scaled_percentage
  else:
    MAX = 1
    scaled_percentage = (y_pred_prob * MAX) / 100
    return scaled_percentage

@st.cache
def __load_and_preprocess_custom_image(image_path):
  img = load_img(image_path, color_mode = 'grayscale', target_size = (700, 700))
  img = img_to_array(img).astype('float32')/255
  return img

@st.cache
def __predict_score(image):
    image = __load_and_preprocess_custom_image(image)
    y_pred = model_eval.predict(np.expand_dims(image, axis=0), verbose=1)[0] 
    y_pred_class = np.argmax(y_pred)
    y_pred_prob = y_pred[y_pred_class]*100 
    score = __calculate_score(y_pred_class, y_pred_prob)
    return y_pred_class, score

@st.cache
def __auto_encode(image):
    org_img = load_img(image, color_mode = 'grayscale')
    org_img = img_to_array(org_img)
    img = org_img.astype('float32')
    img = np.expand_dims(img, axis=0)
    y_pred = np.squeeze(model_auto.predict(img, verbose=1))
    img = cv2.convertScaleAbs(y_pred, alpha=(255.0))
    img = Image.fromarray(img)
    return img

@st.cache
def __get_text_from_image(image):
  text = textract.process(image, method='tesseract', encoding='utf-8')
  text = text.decode('utf8')
  return text

st.title("denoise and evaluate images")
img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

model_eval, model_auto = load_models()

if img_file_buffer is not None:
    org = load_img(img_file_buffer)
    y_pred_class, score = __predict_score(img_file_buffer)
    text = __get_text_from_image(img_file_buffer)

    st.image(org, caption=f"Original", width=700)
    st.write("Predicted class : %s" % (CLASS_IDXS[y_pred_class]))
    st.write("Score : %f" % (score))
    st.write(text)

    img = __auto_encode(img_file_buffer)
    file_object = io.BytesIO()
    img.save(file_object, 'PNG')
    y_pred_class, score = __predict_score(file_object)
    text = __get_text_from_image(file_object)

    st.image(img, caption=f"Processed Image", width=700)
    st.write("Predicted class : %s" % (CLASS_IDXS[y_pred_class]))
    st.write("Score : %f" % (score))
    st.write(text)

else:
    st.write('Please upload single image')

