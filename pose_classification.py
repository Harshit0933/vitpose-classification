import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

def load_npy_file(file_path):
    """
    Load an .npy file and return its content.
    """
    try:
        data = np.load(file_path)
        # Add any preprocessing if needed
        return data
    except Exception as e:
        # Handle exceptions, e.g., corrupted files
        return None

# load the data

def load_data(data_folder, cache_filename=None):
    if cache_filename is not None and os.path.exists(cache_filename):
        # Load data from cache if it exists
        data, labels = joblib.load(cache_filename)
    else:
        data = []
        labels = []

        # Get class names from subdirectories in 'data_folder'
        class_names = sorted(os.listdir(data_folder))
        class_mapping = {class_name: class_id for class_id, class_name in enumerate(class_names)}

        for class_name in class_names:
            class_path = os.path.join(data_folder, class_name)
            class_id = class_mapping[class_name]

            for npy_file in os.listdir(class_path):
                if npy_file.endswith(".npy"):
                    file_path = os.path.join(class_path, npy_file)
                    data_loaded = load_npy_file(file_path)
                    if data_loaded is not None:
                        data.append(data_loaded)
                        labels.append(class_id)

        if cache_filename is not None:
            # Cache the loaded data for future use
            joblib.dump((data, labels), cache_filename)

    return np.array(data), np.array(labels)

if __name__ == "__main__":

    data_dir = "C:/Users/HARSHIT/PycharmProjects/actiom/actions"
    cache_filename = "cached_data.pkl"

    data, labels = load_data(data_dir, cache_filename=cache_filename)

    # Now you can use 'data' and 'labels' in your classification task
    print("Data shape:", data.shape)
    print("Labels shape:", labels.shape)

    # Split the dataset into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=42)

    # 'X_train' and 'y_train' contain the training data and labels, respectively
    # 'X_test' and 'y_test' contain the testing data and labels, respectively

    # Reshape the data to match the model's input shape
    X_train = X_train.reshape(-1, 17, 64, 48)
    X_test = X_test.reshape(-1, 17, 64, 48)

    # Get class names from the directory structure
    class_names = sorted(os.listdir(data_dir))

    #CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(17, 64, 48)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(class_names), activation='softmax')  # Adjust the number of units
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    num_epochs = 10
    batch_size = 32

    history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    model.save("pose_classification_model.h5")
