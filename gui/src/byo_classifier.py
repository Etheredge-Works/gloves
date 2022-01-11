import sys
sys.path.append('gloves')
sys.path.append('.')
import streamlit as st
from gloves import utils
#from gloves import custom_model
#from PIL import Image
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
#import numpy as np
import mlflow
#from pathlib import Path
import datetime
import tensorflow_addons as tfa

#model = custom_model.get_model()

st.write("""
# Roll Your Own Pet Classifier

A demo site for my Siamese Network experiments in image comparisons. More information avaible in end apendix
""")

# TODO dynamically pull from a medium artical with information in it
st.write("""
## How to use
Input an anchor image of the animal type you want to classifer (best on dog/cat breeds).
\n
Input 1 or more other images to see predicting distances.
""")



@st.cache(allow_output_mutation=True)
def load_model(mlflow_model_name='gloves', mlflow_model_stage='Production'):
   client = mlflow.tracking.MlflowClient()
   model_version = client.get_latest_versions(name=mlflow_model_name, stages=[mlflow_model_stage])[0]
   model_path = client.download_artifacts(model_version.run_id, 'model', dst_path='/tmp/')
   model_summary_path = client.download_artifacts(model_version.run_id, 'model_summary.txt', dst_path='/tmp/')
   model = tf.keras.models.load_model(
      model_path,
      custom_objects={'ContrastiveLoss': tfa.losses.ContrastiveLoss})

   with open(model_summary_path) as f:
    summary = f.read()
   return model, model_version, summary


# TODO why does st.cache cause incorecct prediction returns?
@st.cache()
def predict(model, anchor, others):
    # TODO could optimize like I did for nway, but that's too much work right now
    anchors = np.stack([anchor for _ in others])
    #anchors = [anchor for _ in others]
    others = np.stack(others)
    return model.predict([anchors, others])


model, model_version, summary = load_model()
cls_model, cls_model_version, cls_summary = load_model('gloves-classifier')
#st.write(f"{summary}")
#st.write(f"{len(summary)}")

#st.table((
   #('Creation Date',
     #datetime.datetime.fromtimestamp(float(model_version.creation_timestamp/1000)).strftime('%Y-%m-%d %H:%M:%S.%f')),
   #('Model Version', model_version.version)
#))
st.write(f"""
   ### Distance Model Information
   | Model Creation Date |  Model Version |
   | :-------: | :---: |
   | {datetime.datetime.fromtimestamp(float(model_version.creation_timestamp/1000)).strftime('%Y-%m-%d %H:%M:%S.%f')} | {model_version.version} |

   ### Classifier Model Information
   | Model Creation Date |  Model Version |
   | :-------: | :---: |
   | {datetime.datetime.fromtimestamp(float(model_version.creation_timestamp/1000)).strftime('%Y-%m-%d %H:%M:%S.%f')} | {model_version.version} |

""")

#st.slider('Cutoff Threshold', min_value=0.00001)
#help='Value to use for differentiating images.')

anchor_file = st.file_uploader("Input an Image to use an anchor image", type="jpg",)
if anchor_file is not None:
   st.image(anchor_file, caption="Uploaded Anchor Image.", use_column_width=True)

other_files = st.file_uploader("Input Images to compare to Anchor Image", accept_multiple_files=True, type="jpg")
if other_files is not None:
    cols = st.beta_columns(4)
    for idx, file in enumerate(other_files):
        cols[idx%4].image(file, caption=file.name, use_column_width=True)
#st.write(Image.frombytes('RGB', anchor_file))

if anchor_file and other_files:
   #st.write("Reshaping and prepping images...")
   cleaned_anchor = utils.simple_decode(anchor_file.read())
   cleaned_others = [utils.simple_decode(other_file.read()) for other_file in other_files]


   st.write("Predicting on images...")
   # TODO pass all at once
   prediction_values = predict(model, cleaned_anchor, cleaned_others)
   st.write("## Distances")
   cols = st.beta_columns(4)
   for idx, (file, predictions) in enumerate(zip(other_files, prediction_values)):
       cols[idx%4].image(file, caption=str(*predictions), use_column_width=True)

   st.write(f"""## Distances Below theshold
   This is for trying to determine there's a good way to act off the anchor image.
   """)
   threshold = st.number_input('Cutoff Threshold', value=1.0, min_value=0., step=0.01)
   cols = st.beta_columns(4)
   for idx, (file, predictions) in enumerate(zip(other_files, prediction_values)):
       if predictions[0] <= threshold:
            cols[idx%4].image(file, caption=str(*predictions), use_column_width=True)
   
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



   # TODO pull out the closest N matching images from the dataset based on distance

#st.text(prediction_value)
#st.help(prediction_value)
st.write("# Apendix")
st.write("""
## Background
I originally started this project as a way to explore few-shot learners. It has since been a playground for me to explore tons of other technologies like
Kubeflow, DVC, MLflow, Weights and Bias, docker, etc.
""")
st.write("## Model Summary")
model.summary(print_fn=st.text)
sub_model = [layer for layer in model.layers if layer.name == 'model'][0]
st.write("## Sub Model Summary (duplicated siamese network / latent encoder)")
sub_model.summary(print_fn=st.text)
st.write(model)
