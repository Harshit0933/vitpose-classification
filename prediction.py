import os
from tensorflow.keras.models import load_model
import numpy as np

data_dir = "C:/Users/HARSHIT/PycharmProjects/actiom/actions"
# Load the trained model
model = load_model("pose_classification_model.h5")

# Prepare new input data
new_data = np.random.random((1, 17, 64, 48))

# Make predictions
predictions = model.predict(new_data)

# Interpret predictions
predicted_class_ids = np.argmax(predictions, axis=1)
predicted_class_id = predicted_class_ids[0]  # Get the predicted class ID

# Get the corresponding class name based on the class ID from the 'class_names' list
class_names = sorted(os.listdir(data_dir))
predicted_class_name = class_names[predicted_class_id]

print("Predicted Class Name:", predicted_class_name)




