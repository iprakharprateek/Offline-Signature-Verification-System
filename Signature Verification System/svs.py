import tkinter as tk
from tkinter import filedialog
import os
import cv2
import numpy as np
from skimage import io, color
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, auc, roc_curve
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def preprocess_image(image):
    grayscale_image = color.rgb2gray(image)
    # Calculate the Otsu threshold
    otsu_threshold = threshold_otsu(grayscale_image)
    # Convert the grayscale image to binary using the Otsu threshold
    binary_image = grayscale_image > otsu_threshold
    # Crop the signature
    r, c = np.where(binary_image == 0)
    cropped_image = binary_image[r.min() : r.max() + 1, c.min() : c.max() + 1]
    return cropped_image

def extract_features(image):
    # Calculate the height-width ratio
    height, width = image.shape
    height_width_ratio = height / width

    # Calculate the occupancy ratio
    occupancy_ratio = np.sum(image) / (height * width)

    # Calculate the density ratio
    density_ratio = np.sum(image) / image.size

    # Convert the binary image to grayscale
    grayscale_image = image.astype(np.uint8) * 255
    harris_corners = cv2.cornerHarris(grayscale_image, blockSize=2, ksize=3, k=0.04)
    harris_corners = cv2.threshold(harris_corners, 0.01 * harris_corners.max(), 255, 0)[1]
    critical_points = np.argwhere(harris_corners != 0)
    num_critical_points = len(critical_points)

    # Calculate the center of gravity
    y_coords, x_coords = np.where(image > 0)
    center_of_gravity = np.mean(y_coords), np.mean(x_coords)

    # Calculate the slope of center of gravities
    slope_of_center_of_gravities = 0.0
    if len(x_coords) > 1:  # Ensure at least 2 x-coordinates to calculate slope
        y_diff = np.diff(y_coords)
        x_diff = np.diff(x_coords)
        valid_indices = np.where(x_diff != 0)[0]  # Indices where x_diff is not zero
        if len(valid_indices) > 0:  # Check if there are valid indices to avoid divide by zero
            slope_of_center_of_gravities = np.mean(y_diff[valid_indices] / x_diff[valid_indices])

    

    # Ensure that all feature values are finite
    features = [
        height_width_ratio if np.isfinite(height_width_ratio) else 0.0,
        occupancy_ratio if np.isfinite(occupancy_ratio) else 0.0,
        density_ratio if np.isfinite(density_ratio) else 0.0,
        num_critical_points if np.isfinite(num_critical_points) else 0.0,
        center_of_gravity[0] if np.isfinite(center_of_gravity[0]) else 0.0,
        center_of_gravity[1] if np.isfinite(center_of_gravity[1]) else 0.0,
        slope_of_center_of_gravities if np.isfinite(slope_of_center_of_gravities) else 0.0
    ]

    return features

    

# Set the paths for the training and testing datasets
training_dataset_folder = 'E:\Signature Verification System\Training dataset'
test_dataset_folder = 'E:\Signature Verification System\Testing dataset'

def traintest():
# Load training signatures, preprocess them, and extract features
  training_features = []
  training_labels = []
  for file in os.listdir(training_dataset_folder):
    if file.endswith('.jpg'):
        file_path = os.path.join(training_dataset_folder, file)
        image = io.imread(file_path)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Extract features from the preprocessed image
        features = extract_features(processed_image)
        
        # Determine if the signature is original or forged based on the filename
        label = 'Original' if 'original' in file else 'Forged'
        
        # Append the features and label to the training lists
        training_features.append(features)
        training_labels.append(label)
      
        
# Load testing signatures, preprocess them, and extract features
  testing_features = []
  testing_labels = []
  for file in os.listdir(test_dataset_folder):
    if file.endswith('.jpg'):
        file_path = os.path.join(test_dataset_folder, file)
        image = io.imread(file_path)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Extract features from the preprocessed image
        features = extract_features(processed_image)
        
        # Determine if the signature is original or forged based on the filename
        label = 'Original' if 'original' in file else 'Forged'
        
        # Append the features and label to the testing lists
        testing_features.append(features)
        testing_labels.append(label)

# Create the feature matrices X_train and X_test
  X_train = np.array(training_features)
  X_test = np.array(testing_features)

# Create the label vectors y_train and y_test
  y_train = np.array(training_labels)
  y_test = np.array(testing_labels)

# Train and test the SVM model
  svm_model = svm.SVC()
  svm_model.fit(X_train, y_train)
  svm_predictions = svm_model.predict(X_test)
  svm_accuracy = accuracy_score(y_test, svm_predictions)
  svm_train_accuracy = accuracy_score(y_train, svm_model.predict(X_train))
  print("SVM Accuracy: ",svm_accuracy)


# Train and test the ANN model
  ann_model = MLPClassifier()
  ann_model.fit(X_train, y_train)
  ann_predictions = ann_model.predict(X_test)
  ann_accuracy = accuracy_score(y_test, ann_predictions)
  ann_train_accuracy = accuracy_score(y_train, ann_model.predict(X_train))
  print("ANN Accuracy: ",ann_accuracy)

  le = LabelEncoder()
  y_train = le.fit_transform(y_train)
  y_test = le.transform(y_test)
# Train and test the Decision Tree model
  # Train the decision tree model
  dt_model = DecisionTreeClassifier()
  dt_model.fit(X_train, y_train)

# Make predictions on the test set
  dt_predictions = dt_model.predict(X_test)

# Calculate model performance metrics
  dt_accuracy = accuracy_score(y_test, dt_predictions)
  dt_train_accuracy = accuracy_score(y_train, dt_model.predict(X_train))


# Compute the confusion matrix
  tn, fp, fn, tp = confusion_matrix(y_test, dt_predictions).ravel()

# Print the performance metrics
  print(f"Decision Tree Classifier Accuracy: {dt_accuracy:.2f}")
  print(f"Decision Tree Classifier Train Accuracy: {dt_train_accuracy:.2f}")
  print(f"True Positives: {tp}")
  print(f"True Negatives: {tn}")
  print(f"False Positives: {fp}")
  print(f"False Negatives: {fn}")

# Calculate and print precision and recall
  dt_precision = precision_score(y_test, dt_predictions)
  dt_recall = recall_score(y_test, dt_predictions)
  print(f"Precision: {dt_precision:.2f}")
  print(f"Recall: {dt_recall:.2f}")

# Generate the classification report
  print("Classification Report:")
  print(classification_report(y_test, dt_predictions))

# Plot the confusion matrix
  plt.figure(figsize=(8, 6))
  sns.heatmap(confusion_matrix(y_test, dt_predictions), annot=True, cmap='Blues', fmt='g')
  plt.title('Decision Tree Classifier Confusion Matrix')
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.show()

# Plot the ROC curve and calculate the AUC
  fpr, tpr, thresholds = roc_curve(y_test, dt_model.predict_proba(X_test)[:,1])
  roc_auc = auc(fpr, tpr)

  plt.figure(figsize=(8, 6))
  plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Decision Tree Classifier ROC Curve')
  plt.legend(loc='lower right')
  plt.show()
  return dt_model

def validateSignature(input_signature):
    processed_input = preprocess_image(input_signature)

    input_features = extract_features(processed_input)
    dt_model=traintest()
    predicted_label = dt_model.predict([input_features])[0]
    if predicted_label==0:
       predicted_label="Forged"
    else:
       predicted_label="Original"
    op_text = "Features extracted:\n height-width ratio:" + str(input_features[0]) + "\n occupancy ratio:" + str(input_features[1]) + "\n density ratio:" + str(input_features[2]) + "\n number of critical points:" + str(input_features[3]) + "\n center of gravity:[" + str(input_features[4]) + "," + str(input_features[5]) + "]" + "\n slope of center of gravity:" + str(input_features[6])+"\n Predicted Label : "+str(predicted_label)
    return op_text



def browse_file():
    filepath = filedialog.askopenfilename()
    file_entry.delete(0, tk.END)
    file_entry.insert(0, filepath)

def process_input():
    
    file_path = file_entry.get()
    input_signature = io.imread(file_path)
    
    # Process the input 
    output_text= validateSignature(input_signature)
    
    #output part:
    output_textbox.config(state=tk.NORMAL)
    output_textbox.delete(1.0, tk.END)
    output_textbox.insert(tk.END, output_text)
    output_textbox.config(state=tk.DISABLED)
    
    # Display the processed image in the output area
    photo = ImageTk.PhotoImage(Image.fromarray(preprocess_image(input_signature)))
    output_image_label.configure(image=photo)
    output_image_label.image = photo
# Create main window
root = tk.Tk()
root.title("Signature Verification System")



file_label = tk.Label(root, text="Upload Signature File:")
file_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
file_entry = tk.Entry(root)
file_entry = tk.Entry(root, width=50)
file_entry.grid(row=1, column=1, padx=5, pady=5)
browse_button = tk.Button(root, text="Browse", command=browse_file)
browse_button.grid(row=1, column=2, padx=5, pady=5)

# Create output widget
output_textbox = tk.Text(root, height=10, width=100)
output_textbox.grid(row=2, column=0, columnspan=3, padx=5, pady=5)
output_textbox.config(state=tk.DISABLED)

# Create processed image area
output_image_label = tk.Label(root)
output_image_label.grid(row=3, column=0, columnspan=3, padx=5, pady=5)

# Create process button
process_button = tk.Button(root, text="Verify", command=process_input)
process_button.grid(row=3, column=0, columnspan=3, padx=5, pady=5)

# Run the main loop
root.mainloop()
