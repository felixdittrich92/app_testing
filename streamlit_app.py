import streamlit as st

import io
import os
from tempfile import NamedTemporaryFile

import cv2
from PIL import Image
from imutils.perspective import four_point_transform

import pytesseract

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img

CLASS_IDXS = ["not good", "good"]
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata/"

@st.cache(allow_output_mutation=True)
def load_models():
  model_eval = load_model('models/doc_model.h5', compile=False)
  model_auto = load_model('models/auto_model.h5', compile=False)
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
def __preprocessing_handy_image(image_path):
  image = cv2.imread(image_path)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (5,5), 0)
  _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  # find contours and sort for largest contour
  cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
  displayCnt = None

  for c in cnts:
      # perform contour approximation
      peri = cv2.arcLength(c, True)
      approx = cv2.approxPolyDP(c, 0.02 * peri, True)
      if len(approx) == 4:
          displayCnt = approx
          break

  # obtain birds' eye view of image
  image = four_point_transform(image, displayCnt.reshape(4, 2))   
  img = Image.fromarray(image)   
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
  custom_oem_psm_config = r'--oem 3 --psm 6'
  text = pytesseract.image_to_string(Image.open(image), config=custom_oem_psm_config, nice=3, lang='eng+deu')
  return text

def app(image):
  stocks = ["Handy Image Preprocessing", "use Autoencoder", "todo - show timer", "todo - use other ocr"]
  check_boxes = [st.sidebar.checkbox(stock, key=stock) for stock in stocks]
  checked_stocks = [stock for stock, checked in zip(stocks, check_boxes) if checked]

  if "Handy Image Preprocessing" in checked_stocks:
    org = __preprocessing_handy_image(image)
  else:
    org = load_img(image)

  y_pred_class, score = __predict_score(org)
  text = __get_text_from_image(org)

#  display env
#  st.write(os.listdir("/usr/share/tesseract-ocr/4.00/tessdata/"))
#  st.write(os.environ)
  st.subheader('Image')
  st.image(org, caption=f"Original", width=700)
  st.subheader('Predictions')
  st.write("Predicted class : %s" % (CLASS_IDXS[y_pred_class]))
  st.write("Score : %f" % (score))
  st.subheader('Extracted text')
  st.text(text)

  if "use Autoencoder" in checked_stocks:
    img = __auto_encode(image)
    file_object = io.BytesIO()
    img.save(file_object, 'PNG')
    temp_file = NamedTemporaryFile(delete=False)
    temp_file.write(file_object.getvalue())
    y_pred_class, score = __predict_score(temp_file.name)
    text = __get_text_from_image(temp_file.name)

    st.write("------------------------------------------")

    st.subheader('Image')
    st.image(img, caption=f"Processed Image", width=700)
    st.subheader('Predictions')
    st.write("Predicted class : %s" % (CLASS_IDXS[y_pred_class]))
    st.write("Score : %f" % (score))
    st.subheader('Extracted text')
    st.text(text)
  st.markdown("Built with Streamlit by [Felix](https://github.com/felixdittrich92?tab=repositories)")


st.title("denoise and evaluate")
st.title("scanned document images")
img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

model_eval, model_auto = load_models()

if img_file_buffer is not None:
  temp_file = NamedTemporaryFile(delete=False)
  temp_file.write(img_file_buffer.getvalue())
  app(temp_file.name)
else:
  demo = 'images/doc.jpg'
  app(demo)
  st.write("------------------------------------------")
  st.title('Try it and upload your own scanned document !')
