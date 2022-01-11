import sys
sys.path.append('gloves')
sys.path.append('.')
import streamlit as st
from gloves import utils
#from gloves import custom_model
from PIL import Image
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
import numpy as np
import mlflow
from pathlib import Path
import datetime
import tensorflow_addons as tfa

#model = custom_model.get_model()
st.title("Pet Breed Similarity")
st.write("""
A demo site for my Siamese Network experiments in image comparisons.

automated.
""")


# TODO dynamically pull from a medium artical with information in it
st.write("""
## Background
I originally started this project as a way to explore few-shot learners. It has since been a playground for me to explore tons of other technologies like
Kubeflow, DVC, MLflow, Weights and Bias, docker, etc.


## How to use
Simply input an achor image and two other images of animals to compare it to and it'll output the distance between them.
""")

import os


@st.cache(allow_output_mutation=True)
def load_model(mlflow_model_name, mlflow_model_stage='Production'):
   client = mlflow.tracking.MlflowClient()
   model_version = client.get_latest_versions(name=mlflow_model_name, stages=[mlflow_model_stage])[0]
   model_artifact = client.download_artifacts(model_version.run_id, 'model', dst_path='/tmp/')
   model = tf.keras.models.load_model(
      '/tmp/model',
      custom_objects={"ContrastiveLoss": tfa.losses.ContrastiveLoss})
   return model, model_version


# TODO why does st.cache cause incorecct prediction returns?
def dist_predict(model, anchor, other):
   return model.predict([np.expand_dims(anchor, axis=0), np.expand_dims(other, axis=0)])[0][0]

#def class_predict(model, anchor, other):
   #return model.predict([np.expand_dims(anchor, axis=0), np.expand_dims(other, axis=0)])[0][0]

dist_model, dist_model_version = load_model('gloves')
clas_model, clas_model_version = load_model('gloves-classifier')

#st.table((
   #('Creation Date',
     #datetime.datetime.fromtimestamp(float(model_version.creation_timestamp/1000)).strftime('%Y-%m-%d %H:%M:%S.%f')),
   #('Model Version', model_version.version)
#))
st.write(f"""
   ### Distance Model Information
   | Model Creation Date |  Model Version |
   | :-------: | :---: |
   | {datetime.datetime.fromtimestamp(float(clas_model_version.creation_timestamp/1000)).strftime('%Y-%m-%d %H:%M:%S.%f')} | {clas_model_version.version} |

   ### Classifier Model Information
   | Model Creation Date |  Model Version |
   | :-------: | :---: |
   | {datetime.datetime.fromtimestamp(float(clas_model_version.creation_timestamp/1000)).strftime('%Y-%m-%d %H:%M:%S.%f')} | {clas_model_version.version} |

""")

anchor_file = st.file_uploader("Input an Image to use an anchor image", type="jpg",)
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
   st.write("Reshaping and prepping images...")
   cleaned_anchor = utils.simple_decode(anchor_file.read())
   cleaned_other_1 = utils.simple_decode(other_file_1.read())
   cleaned_other_2 = utils.simple_decode(other_file_2.read())

   st.write("Done.")


   st.write("Predicting on images..")
   # TODO pass all at once
   prediction_value_1 = predict(dist_model, cleaned_anchor, cleaned_other_1)
   prediction_value_2 = predict(distmodel, cleaned_anchor, cleaned_other_2)
   prediction_value_3 = predict(dist_model, cleaned_other_1, cleaned_other_2)
   st.write("Done.")
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

   col1.image(anchor_file, caption='Anchor', use_column_width=True)
   col2.image(other_file, caption='Other', use_column_width=True)
   st.write(f"Distance (Anchor vs Other 1): {prediction_value_1}")
   st.write(f"Distance (Anchor vs Other 2): {prediction_value_2}")
   st.write(f"Distance (Other 1 vs Other 2): {prediction_value_3}")
   
   # TODO make one shot classifier with slider option for certainty
   # TODO print model summary
   # TODO print model version
   # TODO add instructions
   # TODO write post about regularizational effectiveness in the application
   # TODO add information on how all figures and charts are auto generated
   # TODO add charts for training information in an appendix
   # TODO implement stratified cross-validation
   # TODO pull in test data for more complete analaysis
   # TODO could just run N random stratified splits to make do cross validation
   # TODO does nways being batch size for batch normalzation improve performance
   # TODO why does test_nway_acc improve when val_loss gets worse
   # TODO add info on tools used like mlflow and model registry
   # TODO add description of project and directions taken and things done like blog pose
   # TODO trim and optmize trained model for deployment

#st.text(prediction_value)
#st.help(prediction_value)
