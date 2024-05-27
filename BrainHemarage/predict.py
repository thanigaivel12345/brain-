from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('best_model.h5')

# Define the target size for resizing images
target_size = (200, 200)

# Function to load and preprocess the image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Function to make predictions
def predict_image(image_path):
    # Preprocess the image.hs
    img_array = preprocess_image(image_path)
    # Make prediction
    prediction = model.predict(img_array)
    # Get the predicted class label
    predicted_class_index = np.argmax(prediction)
    # Map the class index to the class name
    class_labels = ['Brain Hemorrhage', 'Normal' ]
    predicted_class_label = class_labels[predicted_class_index]
    # Get the confidence score
    confidence_score = prediction[0][predicted_class_index]
    return predicted_class_label, confidence_score

# Path to the image you want to predict
image_path = '1.jpg'  # Change this to your image file path

# Make prediction
predicted_class, confidence = predict_image(image_path)

# Print the predicted class name and confidence score
print(f'Predicted Class: {predicted_class}')
print(f'Confidence: {confidence:.2f}')

# Load and display the image
img = image.load_img(image_path)
plt.imshow(img)
plt.axis('off')
plt.title(f'Predicted Class: {predicted_class}\nConfidence: {confidence:.2f}')
plt.show()
