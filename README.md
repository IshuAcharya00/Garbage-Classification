🧠 Image Classification and Real-Time Prediction Web App

Project Overview:

This project is an AI-powered waste classification system that identifies images of waste into six categories — cardboard, glass, metal, paper, plastic, and trash.
It promotes automated waste segregation and supports recycling through an easy-to-use Streamlit web application.


Dataset:

Source: Garbage Classification Dataset (Kaggle)
Classes: cardboard, glass, metal, paper, plastic, trash
Structure: Each class stored in a separate folder containing multiple images.
Diversity: Varying lighting, angles, and backgrounds for robust learning.


Data Preprocessing:

Image Augmentation:
Rotation, Zoom, Horizontal & Vertical Flip, Width/Height Shift

Normalization:
Pixel values scaled to [0,1] and preprocessed using MobileNetV2’s preprocess_input()

Data Split:
80% Training, 20% Validation via flow_from_directory() in Keras


Model Developement:

Base Model: MobileNetV2 (pretrained on ImageNet)
Custom Layers: Global Average Pooling + Dense (Softmax, 6 classes)
Input Size: 224×224×3
Loss Function: categorical_crossentropy
Optimizer: Adam


Trainig Setup:
Epochs: 15
Batch Size: 32
Early Stopping: Custom callback (myCallback) stops training once accuracy > 90%


Model Evaluation :

Metrics: Accuracy, Precision, Recall, F1-score
Visualization: Confusion matrix to analyze misclassifications


Model Saving:
The final model was saved in .keras format for deployment.


Deployment(Streamlit App):

The model was integrated into a Streamlit web app that allows users to:
1)Upload an image
2)Get predicted class and confidence score
3)View prediction probabilities via bar chart
4)Enjoy emoji-based feedback and sidebar instructions


To Run Locally:

pip install -r requiremnts.txt
streamlit run app.py

 
Conclusion:

This project demonstrates a lightweight yet high-performing waste classification system built using MobileNetV2 and transfer learning.


Tech Stack:

Python, TensorFlow / Keras, Streamlit, NumPy, Matplotlib
