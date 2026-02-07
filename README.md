## Dataset
Due to GitHub size limits, the MRI dataset is stored here:
[Download Dataset](https://drive.google.com/drive/folders/1lywrhH21oO8lY29gTu40B0utm2Ditigz?usp=sharing)

🔹 Task 1 – Dataset Preprocessing (COMPLETED)

Raw MRI scans were collected and organized subject-wise. Each scan was mapped to the correct diagnostic label using the provided CSV file.

Standard MRI preprocessing techniques were applied, including skull stripping, normalization, and resizing to ensure consistent model input dimensions.

The dataset was then split into training, validation, and testing sets, ensuring class balance and data integrity.

The output is a clean, structured dataset ready for deep learning model training.

🔹 Task 2 – Binary Classification: CN vs AD (COMPLETED)

A CNN-based deep learning model was built to distinguish Cognitively Normal individuals from those with Alzheimer’s Disease using MRI data.

The model extracts spatial brain features automatically and learns disease-related structural patterns.

Evaluation metrics include Balanced Accuracy, ROC-AUC, F1-Score, Precision, Recall, and Confusion Matrix.

This task validates whether MRI alone can effectively identify Alzheimer’s Disease.

🔹 Task 3 – Multi-Class Classification: CN vs MCI vs AD (COMPLETED)

A multi-class CNN model was developed to detect disease progression stages.

The model differentiates between normal cognition, mild impairment, and advanced Alzheimer’s Disease.

This task is more challenging because MCI features are subtle and overlap with normal aging.

Performance is evaluated using macro F1-score, ROC curves, and class-wise precision & recall.

🔹 Task 4 – Web Application Development (IN PROGRESS)

A web-based interface is being built to provide end-to-end neurological screening.

Users can upload MRI scans, and the system automatically preprocesses images and runs predictions.

The interface displays predicted class and confidence score in an easy-to-understand format.

The goal is to make the tool usable by healthcare workers without AI expertise.

🛠️ Technologies Used

Python

TensorFlow / PyTorch

OpenCV & NiBabel (MRI handling)

Scikit-learn (metrics)

Flask / Streamlit (Web App)

NumPy, Pandas, Matplotlib
