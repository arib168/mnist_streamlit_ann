import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
import cv2
model = keras.models.load_model('model_digit.hdf5',compile=False)#Boolean, whether to compile the model after loading.

st.title("MNIST Image Classification using ANN")
st.subheader("Draw the handwritten digit in the below given space and click on PREDICT")
col1, col2 = st.beta_columns(2)

#with statement in Python is used in exception handling to make the code cleaner and much more readable. 
#It simplifies the management of common resources like file streams.
#to give column 1 and column 2 side by side (if no with col1 box will be up, col2 box will be down)
with col1:  #col1 givs original image
  st.write('Original Image')
  canvas_result = st_canvas(fill_color='#000000',stroke_width=20,stroke_color='#2121b0',
      background_color='#000000',width=225,height=225,drawing_mode="freedraw")

with col2:
  if canvas_result.image_data is not None: #means if there is something drawn then, follow next steps
      img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28)) #resize to 28 *28 
      rescaled = cv2.resize(img, (225, 225)) #zoom in#it is blurred now
      st.write('Rescaled Image')
      st.image(rescaled)

if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# Converting color to gray
    test_x = np.expand_dims(test_x,axis=0) # Flattening of image
    val = model.predict(test_x)
    st.title(f'Result: {np.argmax(val[0])}') # prints the value with highest probability index
    st.bar_chart(val[0])