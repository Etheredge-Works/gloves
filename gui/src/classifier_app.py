import sys
sys.path.append('gloves')
sys.path.append('.')
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
import joblib
from icecream import ic
import utils
import pandas as pd
import datetime

#model = custom_model.get_model()

st.title("""
Pet Breed Classifier

This is a demo site for my Siamese Network experiments in image comparisons.
This page demonstrates the effectiveness of the current siamese network encoder applied to feature extraction for pet breed classifiction on the Oxford pet dataset. 
Process: 
1. A siamese network is trained on the Oxford pet dataset to generate encodings that are close in distance for similar breeds.
2. The encoder is ripped out and used as feature extractor with a dense network tacke don the end.
3. This is compared to two other models. 
   a. Imagenet feature extractor is used with same dense network architecture tacked on. (how does it compare to a bigger, more general feature extractor)
   b. Siamese network architecutre is used for feature extraction with same dense network architecture but is randomly initialized instead of being frozen. (measures effectivness of network pre-training)
""")


@st.cache(allow_output_mutation=True)
def load_models(model_name, mlflow_model_stage='Production'):
   client = mlflow.tracking.MlflowClient()
   model_version = client.get_latest_versions(name=f"gloves_{model_name}", stages=[mlflow_model_stage])[0]


   #st.write(client.get_run(model_version.run_id.mlflow.parentRunId))
   ic('--------------')
   rundata = client.get_run(model_version.run_id)
   parent_id = rundata.data.tags['mlflow.parentRunId']
   parentrundata = client.get_run(parent_id)
   ic(parentrundata)
   model_path = client.download_artifacts(model_version.run_id, model_name, dst_path='/tmp/')
   model = tf.keras.models.load_model(f"{model_path}/data/model.h5")

   label_encoder_path = client.download_artifacts(model_version.run_id, 'label_encoder', dst_path='/tmp/')
   label_encoder = joblib.load(label_encoder_path)
   
   return model, label_encoder, model_version


#def predict(model, anchor, other):
   #return model.predict([np.expand_dims(anchor, axis=0), np.expand_dims(other, axis=0)])[0][0]
model_names = [
   "encoder_frozen",
   "encoder_unfrozen",
   "imagenet_frozen",
   "imagenet_unfrozen",
]
models = [(model_name, load_models(model_name)) for model_name in model_names]


#model, le = load_models()
image = st.file_uploader("Input Image to classify", type="jpg")
if image is not None:
   st.image(image, caption="Uploaded Image.", use_column_width=True)
n = st.number_input("Top Labels to get", value=5, max_value=32, step=1)

cols = st.beta_columns(2)

st.write("# Apendix")
if image is not None:
   data = utils.simple_decode(image.read())
   for idx, (name, (model, le, model_version)) in enumerate(models):
      y_hat = model.predict([np.expand_dims(data, axis=0)])[0]
      sorted_predictions = np.argsort(y_hat)[::-1]
      predictions = le.inverse_transform(sorted_predictions)
      #cols[idx].write(f"{name}\n\n{predictions[:n]}")
      df = pd.DataFrame({
         "Breed": predictions[:n],
         "Confidence": y_hat[sorted_predictions[n]]})
      
      #df.set_index('Breed', inplace=True)

      cols[idx%2].write(name)
      cols[idx%2].dataframe(df)
      with st.beta_expander(f"{name} model info"):
         st.write(f"""
            | Model Creation Date |  Model Version |
            | :-------: | :---: |
            | {datetime.datetime.fromtimestamp(float(model_version.creation_timestamp/1000)).strftime('%Y-%m-%d %H:%M:%S.%f')} | {model_version.version} |
            ## Summary
         """)
         model.summary(print_fn=st.text)

st.write("""
# General Notes/TODOs
- Validation accuracy isn't perfect as there's a chance validation samples were leaked during training of siamese network
- K fold (or another variant) of cross-validation is needed. Currently that's non-trivial to implement using tf.data.Dataset.
   - maybe don't use tf.data.Datase.

""")

#st.text(prediction_value)
#st.help(prediction_value)
