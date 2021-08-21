import streamlit as st
import numpy as np
import tensorflow as tf 
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from PIL import Image, ImageOps 
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.header('Casava Project')

selection = st.sidebar.radio('Navigation', ['Playground', 'Try an Image'])



dataset, info = tfds.load('cassava', with_info= True)

# Extend the cassava dataset classes with 'unknown'
class_names = info.features['label'].names + ['unknown']

# Map the class names to human readable names
name_map = dict(
    cmd='Mosaic Disease',
    cbb='Bacterial Blight',
    cgm='Green Mite',
    cbsd='Brown Streak Disease',
    healthy='Healthy',
    unknown='Unknown')

def preprocess_fn(data):
  image = data['image']

  # Normalize [0, 255] to [0, 1]
  image = tf.cast(image, tf.float32)
  image = image / 255.

  # Resize the images to 224 x 224
  image = tf.image.resize(image, (224, 224))

  data['image'] = image
  return data

def preprocess_image_uploads(image):

  # Normalize [0, 255] to [0, 1]
  image = tf.cast(image, tf.float32)
  image = image / 255.
  # Resize the images to 224 x 224
  image = tf.image.resize(image, (224, 224))
  return image

def plot(examples, predictions=None):
   # Get the images, labels, and optionally predictions
   images = examples['image']
   labels = examples['label']
   batch_size = len(images)
   if predictions is None:
     predictions = batch_size * [None]
   # Configure the layout of the grid
   x = np.ceil(np.sqrt(batch_size))
   y = np.ceil(batch_size / x)
   fig = plt.figure(figsize=(x * 6, y * 7))
   for i, (image, label, prediction) in enumerate(zip(images, labels, predictions)):
     # Render the image
     ax = fig.add_subplot(x, y, i+1)
     ax.imshow(image, aspect='auto')
     ax.grid(False)
     ax.set_xticks([])
     ax.set_yticks([])
     # Display the label and optionally prediction
     x_label = 'Label: ' + name_map[class_names[label]]
     if prediction is not None:
       x_label = 'Prediction: ' + name_map[class_names[prediction]] + '\n' + x_label
       ax.xaxis.label.set_color('green' if label == prediction else 'red')
     ax.set_xlabel(x_label)
   plt.show()


batch = dataset['validation'].map(preprocess_fn).batch(25).as_numpy_iterator()
examples = next(batch)

if selection == 'About':
    pass

if selection == 'Playground':
    st.header('Learn a bit more about the datasets and model')
    st.subheader('Datasets')
    st.write(info.description)
    st.write('Click on the link', info.homepage, 'to read more about the datasets')
    # st.subheader('Sample images in our datasets')
    if st.button('View Sample images'):
        with st.spinner('Please wait while image renders...'):
            plot(examples)
            st.pyplot(plot(examples))



if selection == 'Try an Image':
    image_uploaded = st.sidebar.file_uploader('Upload an image')

    if image_uploaded is not None:
        image = Image.open(image_uploaded)
        st.image(image, width = 350)
        col1, col2 = st.columns(2)
        select_classifier = col1.selectbox('Choose Model', ['CropNet', 'Custom Model'])
        run_classifier = col1.button('Run Disease Classifcation')

  
        img = preprocess_image_uploads(image)
        img = np.array(img)
        imag_dict = dict()
        imag_dict['image'] = img
        imag_dict['image/filename'] = np.array(['user_uploaded'], dtype= 'object')
        imag_dict['label'] = np.array([5])
        imag_dict['image'] = imag_dict['image'].reshape(1,224,224,3)
      

        if run_classifier:
          model = hub.KerasLayer('https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2')
      
          probabilities = model(imag_dict['image'])
          predictions = tf.argmax(probabilities, axis=-1)
          #st.pyplot(plot(imag_dict, predictions))
          metric = tf.keras.metrics.Accuracy()
          labels = imag_dict['label']
          metric.update_state(labels, predictions)
          st.write('Prediction: This leave is likely to be suffering from {}'.format(name_map[class_names[predictions.numpy()[0]]]))
          # st.write(model.get_config())
          # st.write('Accuracy on %s: %.2f' % ('User Uploaded Image', metric.result().numpy()))



    