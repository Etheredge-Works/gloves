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
Siamese Network Exploration

This is a demo site for my Siamese Network experiments in image comparisons. All models were trained using the Oxford pet dataset.

While results perform great on the Oxford pet dataset, not much testing has been done on other datasets. What little testing has been done is not promising. So don't expect great results!

There are 2 sections.

1. Distances and Classifications
    - User entered images are compared
2. Encoder Applied Demo
    - Demos applying the encoders to classification.
    - This is done by comparing the pretrained and encoder vs imagnet models 

""")
@st.cache(allow_output_mutation=True)
def load_model_gloves(mlflow_model_name='gloves', mlflow_model_stage='Production'):
    client = mlflow.tracking.MlflowClient()
    model_version = client.get_latest_versions(name=mlflow_model_name, stages=[mlflow_model_stage])[0]
    model_path = client.download_artifacts(model_version.run_id, 'model', dst_path='/tmp/')
    model_summary_path = client.download_artifacts(model_version.run_id, 'model_summary.txt', dst_path='/tmp/')
    model = tf.keras.models.load_model(model_path)
    with open(model_summary_path) as f:
        summary = f.read()
    return model, model_version, summary

def predict(model, anchor, others):
    # TODO could optimize like I did for nway, but that's too much work right now
    anchors = np.stack([anchor for _ in others])
    #anchors = [anchor for _ in others]
    others = np.stack(others)
    return model.predict([anchors, others])


model, model_version, summary = load_model_gloves()
with st.beta_expander("1. Distances and Classification"):
    # TODO dynamically pull from a medium artical with information in it
    st.write("""
    ## How to use
    Input an anchor image of the animal type you want to classifer (best on dog/cat breeds).
    \n
    Input 1 or more other images to see predicting distances and predicted matching percentage.
    """)

    cls_model, cls_model_version, cls_summary = load_model_gloves('gloves-classifier')
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

    anchor_file = st.file_uploader("Input an Image to use an anchor image", type="jpg",)
    if anchor_file is not None:
        st.image(anchor_file, caption="Uploaded Anchor Image.", use_column_width=True)

    other_files = st.file_uploader("Input Images to compare to Anchor Image", accept_multiple_files=True, type="jpg")
    if other_files is not None:
        cols = st.beta_columns(4)
        for idx, file in enumerate(other_files):
            cols[idx%4].image(file, caption=file.name, use_column_width=True)

    if anchor_file and other_files:
        cleaned_anchor = utils.simple_decode(anchor_file.read())
        cleaned_others = [utils.simple_decode(other_file.read()) for other_file in other_files]

        # TODO pass all at once
        dist_prediction_values = predict(model, cleaned_anchor, cleaned_others)
        cls_prediction_values = predict(cls_model, cleaned_anchor, cleaned_others)
        st.write("## Distances")
        cols = st.beta_columns(4)
        for idx, (file, predictions, matches) in enumerate(zip(other_files, dist_prediction_values, cls_prediction_values)):
            dist = predictions[0]
            y_hat = matches[0]
            cols[idx%4].image(file, caption=f"Dist: {dist}    match: {y_hat}", use_column_width=True)

with st.beta_expander("1.1: Model Summary"):
    st.write("## Model Summary")
    model.summary(print_fn=st.text)
    sub_model = [layer for layer in model.layers if layer.name == 'model'][0]
    st.write("## Sub Model Summary (duplicated siamese network / latent encoder)")
    sub_model.summary(print_fn=st.text)
    st.write(model)

@st.cache(allow_output_mutation=True)
def load_model(model_name, mlflow_model_stage='Production'):
   client = mlflow.tracking.MlflowClient()
   model_version = client.get_latest_versions(name=f"{model_name}", stages=[mlflow_model_stage])[0]
   rundata = client.get_run(model_version.run_id)
   #parent_id = rundata.data.tags['mlflow.parentRunId']
   model_path = client.download_artifacts(model_version.run_id, model_name, dst_path='/tmp/')
   model = tf.keras.models.load_model(f"{model_path}")

   label_encoder_path = client.download_artifacts(model_version.run_id, f'{model_name}_label_encoder.joblib', dst_path='/tmp/')
   label_encoder = joblib.load(label_encoder_path)
   
   return model, label_encoder, model_version, rundata


with st.beta_expander("2. Encoder Applied Demos"):
    st.write("""
        This page demonstrates the effectiveness of the current siamese network encoder applied to feature extraction for pet breed classifiction on the Oxford pet dataset. 
        Process: 

        1. A siamese network is trained on the Oxford pet dataset to generate encodings that are close in distance for similar breeds.
        2. The encoder is ripped out and used as feature extractor with a dense network tacked on the end.
        3. This is compared to Three other models. 
            - Imagenet feature extractor is used with same dense network architecture tacked on. (how does it compare to a bigger, more general feature extractor)
            - Same imagenet but unfrozen
            - Siamese network architecture is used for feature extraction with same dense network architecture but is randomly initialized instead of being frozen. (measures effectivness of network pre-training)
    """)

    model_names = [
        "gloves_encoder_frozen",
        "gloves_encoder_unfrozen",
        "gloves_imagenet_frozen",
        "gloves_imagenet_unfrozen",
    ]
    models = [(model_name, load_model(model_name)) for model_name in model_names]

    image = st.file_uploader("Input Image to classify", type="jpg")
    if image is not None:
        st.image(image, caption="Uploaded Image.", use_column_width=True)
    n = st.number_input("Top Labels to get", value=5, max_value=34, step=1)

    multi_cols = st.beta_columns(2)

    if image is not None:
        data = utils.simple_decode(image.read())
        for idx, (name, (model, le, model_version, rundata)) in enumerate(models):
            y_hat = model.predict([np.expand_dims(data, axis=0)])[0]
            sorted_predictions = np.argsort(y_hat)[::-1]
            predictions = le.inverse_transform(sorted_predictions)
            #cols[idx].write(f"{name}\n\n{predictions[:n]}")
            df = pd.DataFrame({
                "Breed": predictions[:n],
                "Confidence": y_hat[sorted_predictions[:n]]})
            
            multi_cols[idx%2].write(name)
            multi_cols[idx%2].dataframe(df)
    

    st.write("""
    ### General Notes/TODOs
    - Validation accuracy isn't perfect as there's a chance validation samples were leaked during training of siamese network
    - K fold (or another variant) of cross-validation is needed. Currently that's non-trivial to implement using tf.data.Dataset.
    - maybe don't use tf.data.Dataset.
    - Could auto-update models periodically, but that might mess with storage and streamlit caching. 
    - Currently setting for just getting the latest model every reboot

    """)

    #st.text(prediction_value)
for idx, (name, (model, le, model_version, rundata)) in enumerate(models):
    with st.beta_expander(f"2.{idx+1}: {name} model info"):
        st.write(f"""
            | Model Creation Date |  Model Version |
            | :-------: | :---: |
            | {datetime.datetime.fromtimestamp(float(model_version.creation_timestamp/1000)).strftime('%Y-%m-%d %H:%M:%S.%f')} | {model_version.version} |
        """)
        st.write(rundata.data.metrics)
        st.write("## Summary")
        model.summary(print_fn=st.text)
#st.help(prediction_value)
