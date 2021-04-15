import streamlit as st
from gloves import utils
from gloves.main import NormDistanceLayer
#from gloves import custom_model
from PIL import Image
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
import numpy as np
import mlflow
from pathlib import Path

#model = custom_model.get_model()

st.title("""
Pet Bread Similarity

This is a demo site for my Siamese Network experiment in image comparisons.
""")

client = mlflow.tracking.MlflowClient()
MODEL_NAME='gloves'
MODEL_STAGE='Production'
#model = mlflow.keras.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
model_version = client.get_latest_versions(name=MODEL_NAME, stages=[MODEL_STAGE])[0]

model_artifact = client.download_artifacts(model_version.run_id, 'model', dst_path='/tmp/')
model = tf.keras.models.load_model('/tmp/model')

st.cache()
anchor_file = st.file_uploader("Input an Image", type="jpg",)
if anchor_file is not None:
   st.image(anchor_file, caption="Uploaded Anchor Image.", use_column_width=True)

other_file_1 = st.file_uploader("Input Image 1 to compare to Anchor Image", type="jpg")
if other_file_1 is not None:
   st.image(other_file_1, caption="Uploaded Other Image 1.", use_column_width=True)

other_file_2 = st.file_uploader("Input Image 2 to compare to Anchor Image", type="jpg")
if other_file_2 is not None:
   st.image(other_file_2, caption="Uploaded Other Image 2.", use_column_width=True)
#st.write(Image.frombytes('RGB', anchor_file))

if anchor_file and other_file_1 and other_file_2:
   st.write("Processing Images...")
   cleaned_anchor = utils.simple_decode(anchor_file.read())
   cleaned_other_1 = utils.simple_decode(other_file_1.read())
   cleaned_other_2 = utils.simple_decode(other_file_2.read())

   #cleaned_anchor = utils.decode_img(tf.io.decode_raw(anchor_file, 'uint8'))
   #cleaned_other = utils.decode_img(tf.io.decode_raw(anchor_file, 'uint8'))
   st.write("Done.")

   #prediction_value = st.empty()

   #st.write(f"Predicion: {prediction}")
   prediction_value_1 = model.predict([np.expand_dims(cleaned_anchor, axis=0), np.expand_dims(cleaned_other_1, axis=0)])[0][0]
   prediction_value_2 = model.predict([np.expand_dims(cleaned_anchor, axis=0), np.expand_dims(cleaned_other_2, axis=0)])[0][0]
   col1, col2 = st.beta_columns(2)
   if prediction_value_1 < prediction_value_2:
      st.write("Image 1 is closer to anchor")
      #st.image([anchor_file, other_file_1], caption=['Anchor', 'Other'])
      #other_file_2 = Image.open(other_file_2).c
      other_file = other_file_1
   else:
      st.write("Image 2 is closer to anchor")
      #st.image([anchor_file, other_file_2], caption=['Anchor', 'Other'], use_column_width=True)
      other_file = other_file_2

   col1.image(anchor_file, caption='Anchor')
   col2.image(other_file_1, caption='Match')
   #col3.image(other_file_2, caption='Other 2')
   #if prediction_value < 0.5:
      #st.write("These are NOT the same animal.")
   #else:
      #st.write("These are the same animal.")
   st.write(f"Prediction value_1: {prediction_value_2}")
   st.write(f"Prediction value_2: {prediction_value_2}")
   
   #prediction_value.text(model.predict([np.expand_dims(cleaned_anchor, axis=0), np.expand_dims(cleaned_other, axis=0)]))
   #st.write(model.predict([np.expand_dims(cleaned_anchor, axis=0), np.expand_dims(cleaned_other, axis=0)]))

#st.text(prediction_value)
#st.help(prediction_value)
