# GWU_DNSC_3288_Project
Digit Recognizer. Introduction to Deep Learning, Neural Networks, and Pattern Recognition.

## Basic Information
- **Person or organization developing model**: Jonathan, jon@yahoo.com
- **Model date**: November, 2024
- **Model version**: 1.0
- **License**: MIT *?*
- **Model implementation code**: https://colab.research.google.com/drive/1KXlQ71soA8vGusOqcWzRjidf3ikVfA7K?usp=drive_link *?*

## Intended Use
- **Primary intended uses**: This model is an *example* multiclass digit classifier, incorporating various deep learning techniques from the Keras library. The techniques closely followed Kaggle's *Deep Learning Course*, and a handful of tutorials; notably, the *Deep neural network the Keras way* and *Simple deep MLP with Keras* tutorials (all found on the 'Overview' tab of the Digit Recognizer kaggle page).
- **Primary intended users**: Professor Hall and colleagues in GWU DNSC 3288.
- **Out-of-scope use cases**: Any use beyond GWU DNSC 3288, as an educational example, is out-of-scope.

## Training data
- **Source of training data**: https://www.kaggle.com/c/digit-recognizer/data found on GWU Blackboard. Email jphall@gwu.edu for more information.
- **How training data was divided into training and validation data**: The training data was divided into training and validation using the *train_test_split* function from the *model_selection* module of the *scikit-learn* library. The *test_size* parameter was set to 0.1 so that 10% of the 42,000 rows in the *training.csv* would be set aside for validation and the remaining 90% of the data would be use for fitting the model. Furthermore, the *random_state* parameter was set to 0 for reproducibility.
- **Number of rows in training and validation data**:
  -   Training rows: 37,800
  -   Validation rows: 4,200
-   **Data dictionary**:
-   


