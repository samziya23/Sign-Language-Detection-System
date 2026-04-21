# Sign Language Model

Place your trained Keras model here as:
  sign_model.h5       ← Keras HDF5 format (recommended)
  sign_model/         ← TensorFlow SavedModel directory format

Expected input shape:  (None, 64, 64, 3)   float32  values in [0, 1]
Expected output shape: (None, 29)           softmax probabilities

Class order (index → label):
  0-25: A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
  26: del
  27: nothing
  28: space

If no model is found at startup, the app runs in Demo Mode.
