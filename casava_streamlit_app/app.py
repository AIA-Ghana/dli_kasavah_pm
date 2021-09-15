import streamlit as st
import numpy as np
import tensorflow as tf 
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from PIL import Image, ImageOps 
import matplotlib.pyplot as plt
import cv2 
from efficientnet.tfkeras import EfficientNetB3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout  
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam,SGD,RMSprop,Adamax
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold

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

def my_init(shape, dtype=None):
    initializer = tf.keras.initializers.he_uniform(seed = 1)
    return initializer(shape, dtype=dtype)

def create_custom_model():
  TARGET_SIZE = 224
  base_model = EfficientNetB3(weights = 'imagenet', include_top=False, input_shape = (TARGET_SIZE, TARGET_SIZE, 3), pooling=None)
  base_output = base_model.output
  pooling_layer = layers.GlobalAveragePooling2D()(base_output)
  Dense1 = layers.Dense(256, activation = "relu", kernel_initializer=my_init)(pooling_layer)
  BN1 = layers.BatchNormalization()(Dense1)
  dropout = layers.Dropout(0.2)(BN1)
  model = layers.Dense(5, activation="softmax")(dropout)

  model = models.Model(base_model.input, model)

  model.compile(optimizer = 'adam', 
                loss = "sparse_categorical_crossentropy", 
                metrics=["acc"])

  return model

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
        test_image = Image.open(image_uploaded)
        st.image(test_image, width = 350)
        col1, col2 = st.columns(2)
        select_classifier = col1.selectbox('Choose Model', ['All Models', 'Custom Model'])
        run_classifier = col1.button('Run Disease Classifcation')

  
        img = preprocess_image_uploads(test_image)
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
          metric = tf.keras.metrics.Accuracy()
          labels = imag_dict['label']
          predicted_class = name_map[class_names[predictions.numpy()[0]]]
          st.write('CropNet Prediction: {} Dectected'.format(predicted_class))

          #####Custom Model
          model = create_custom_model()
          model.load_weights("cassava.h5")
          im = []
          #test_image = Image.open(image_uploaded)
          img_array = np.array(test_image)
          image_from_array = Image.fromarray(img_array, 'RGB')
          size_image = image_from_array.resize((224, 224))
          im.append(np.array(size_image))
          fv=np.array(im)
          fv = fv.astype('float32')/255
          prediction = model.predict(fv)
          predictions = tf.argmax(prediction, axis=-1)
          predicted_class = name_map[class_names[predictions.numpy()[0]]]
          confidence = round(float(max(prediction[0])), 2) * 100
          st.write('Custom Model Prediction: {} Detected with {}% confidence'.format(predicted_class, confidence  ) ) 

        # elif run_classifier and select_classifier == 'Custom Model':
        #   model = create_custom_model()
        #   model.load_weights("cassava.h5")
        #   im = []
        #   test_image = Image.open(image_uploaded)
        #   img_array = np.array(test_image)
        #   #cv2.imwrite('out.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

        #   image_from_array = Image.fromarray(img_array, 'RGB')
        #   size_image = image_from_array.resize((224, 224))
        #   im.append(np.array(size_image))
        #   fv=np.array(im)
        #   fv = fv.astype('float32')/255
        #   prediction = model.predict(fv)
        #   predictions = tf.argmax(prediction, axis=-1)
        #   predicted_class = name_map[class_names[predictions.numpy()[0]]]
        #   confidence = round(float(max(prediction[0])), 2) * 100
        #   st.write('Prediction: This leaf is likely to be suffering from {} with {}% confidence'.format(predicted_class, confidence  ) ) 


    