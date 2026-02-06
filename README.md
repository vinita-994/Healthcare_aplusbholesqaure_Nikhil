This project focuses on early detection of neurological disorders using structural brain MRI data.  
It targets Alzheimer’s Disease and Mild Cognitive Impairment through AI-based medical image analysis.  
The system is fully software-driven and supports screening in resource-limited healthcare environments.  
MRI scans are paired with subject-level clinical labels such as CN, MCI, and AD.  


Task-1 preprocessing has been successfully completed for the entire MRI dataset.  
Raw DICOM scans are converted into standardized 3D brain volumes for analysis.  
The pipeline performs skull stripping, MNI normalization, and grey matter segmentation.  
Intensity normalization and resizing ensure consistent inputs for deep learning models.  
Original raw MRI data remains untouched during all preprocessing steps.  


This stage produces a clean, structured, and model-ready neuroimaging dataset.  
Upcoming tasks will implement advanced 3D CNN architectures for disease classification.  
Training strategies will focus on high accuracy, robustness, and clinical reliability.  
The goal is a medical-grade AI system achieving over 96% diagnostic accuracy.  
EOF
