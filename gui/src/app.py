import streamlit as st
from gloves import utils
from gloves import custom_model
from PIL import Image
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
import numpy as np

model = custom_model.get_model()

st.title("Testing")



anchor_file = st.file_uploader("Input an Image", type="jpg",)
st.image(anchor_file)
other_file = st.file_uploader("Input another Image", type="jpg")
st.image(other_file)
#st.write(Image.frombytes('RGB', anchor_file))

st.write("Processing Images...")
cleaned_anchor = utils.simple_decode(anchor_file.read())
cleaned_other = utils.simple_decode(other_file.read())

#cleaned_anchor = utils.decode_img(tf.io.decode_raw(anchor_file, 'uint8'))
#cleaned_other = utils.decode_img(tf.io.decode_raw(anchor_file, 'uint8'))
st.write("Done.")

st.write("Prediction")
prediction_value = st.empty()


#st.write(f"Predicion: {prediction}")
prediction_value.text(model.predict([np.expand_dims(cleaned_anchor, axis=0), np.expand_dims(cleaned_other, axis=0)]))
#st.write(model.predict([np.expand_dims(cleaned_anchor, axis=0), np.expand_dims(cleaned_other, axis=0)]))

#st.text(prediction_value)
#st.help(prediction_value)
