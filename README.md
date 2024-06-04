# K-Nearest-Neighbors-KNN-Supervised-Learning-Classification-Example-Template-Code

Original code came from Module 3, Classification, of IBM's Machine Learning with Python Coursera course (https://www.coursera.org/learn/machine-learning-with-python#modules).

What I added to the project:
- Created ten rows of new data
- Update KNN model to predict 'custcat' for new data
- Export results to CSV
- Plot data with two types of 3d plots

Use this code to train and test KNN models on your own data as well as create new data to forecast classification categories and check model accuracy.

# Project deliverables:

Initial data

![initial_custcat_data_for_KNN_model](https://github.com/NollieAnalysis/K-Nearest-Neighbors-KNN-Supervised-Learning-Classification-Example-Template-Code/assets/163913188/75ac3f16-7c52-4e2d-b694-5d6627bd96ba)

Use sklearn's preprocessing.StandardScaler to normalize data

![preprocessing-standardscaler-to-normalize-data](https://github.com/NollieAnalysis/K-Nearest-Neighbors-KNN-Supervised-Learning-Classification-Example-Template-Code/assets/163913188/d8723b08-8a68-4e7b-85bc-f152e60fe207)

Train and test set accuracy with initial k value (initial value was 7)

![initial-train-test-set-accuracy-with-k-chosen-as-7](https://github.com/NollieAnalysis/K-Nearest-Neighbors-KNN-Supervised-Learning-Classification-Example-Template-Code/assets/163913188/7a9ef134-bc36-47ec-80dc-1bd732d72c27)

Choosing the best k value

![choosing-best-k-value](https://github.com/NollieAnalysis/K-Nearest-Neighbors-KNN-Supervised-Learning-Classification-Example-Template-Code/assets/163913188/9cc9e9fd-6202-4e35-916d-18bc01434dbb)

Creating 10 rows of new data (matching parameters of old data) for custcat classification

![new-data-for-knn-model](https://github.com/NollieAnalysis/K-Nearest-Neighbors-KNN-Supervised-Learning-Classification-Example-Template-Code/assets/163913188/1d184d62-bb9c-42bc-97f0-a0f4d406867d)

Adding a new column ('custcat_predicted') for custcat classification for each new data point and saving results as a CSV file

![new_data_with_custcat_predicted_column_added_and_exported_to_CSV_KNN](https://github.com/NollieAnalysis/K-Nearest-Neighbors-KNN-Supervised-Learning-Classification-Example-Template-Code/assets/163913188/a2907508-1369-401a-8a47-c0ceefe9eee1)
