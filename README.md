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
- **Source of training data**: GWU Blackboard. Email jphall@gwu.edu for more information.
- **How training data was divided into training and validation data**: The training data was divided into training and validation using the *train_test_split* function from the *model_selection* module of the *scikit-learn* library. The *test_size* parameter was set to 0.1 so that 10% of the 42,000 rows in the *training.csv* would be set aside for validation and the remaining 90% of the data would be use for fitting the model. Furthermore, the *random_state* parameter was set to 0 for reproducibility.
- **Number of rows in training and validation data**:
  -   Training rows: 37,800
  -   Validation rows: 4,200
-   **Data dictionary**:

| Name       | Modeling Role       | Measurement Level       | Description       |
|----------------|----------------|----------------|----------------|
| label  | Actual value of the digit  | int  | The digit that was drawn by the user. Each digit is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total (i.e., 784 total pixel columns)  |
| pixel0  | Single pixel-value of associated pixel. | int  | A value between 0 and 255, inclusive. Value of pixel indicates the lightness or darkness of that pixel, with higher numbers meaning darker. |
| pixel1  | Single pixel-value of associated pixel.  | int  | A value between 0 and 255, inclusive. Value of pixel indicates the lightness or darkness of that pixel, with higher numbers meaning darker.  |
| ...  | ...  | ...  | ... |
| pixel783  | Single pixel-value of associated pixel  | int  | A value between 0 and 255, inclusive. Value of pixel indicates the lightness or darkness of that pixel, with higher numbers meaning darker. |

## Test Data
- **Source of test data**: GWU Blackboard. Email jphall@gwu.edu for more information.
- **Number of rows in test data**: 28,000
- **Differences in columns between training and test data**: Test data did not contain a *label* column.

## Model details
- **Columns used as inputs in the final model**: All pixel columns ('pixel0-pixel783') used for fitting and validation to minimize error.
- **Column used as target in the final model**: 'Label'
- **Type of model**: Artificial Neural Network
- **Software used to implement the model**: Python, Pandas, Matplotlib, NumPy, Scikit-Learn, TensorFlow, Keras
- **Version of the modeling software**:
  - Python: 3.10.12
  - Pandas: 2.2.2
  - Matplotlib: 3.8.0
  - NumPy: 1.26.4
  - Scikit-Learn: 1.5.2
  - TensorFlow: 2.17.0
  - Keras: 3.4.1
- **Hyperparameters and other setting of model**:
```
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimum amount of change to count as an improvement
    patience=10, # how many epochs to wait before stopping
    restore_best_weights=True
)

model = Sequential([
    Input([input_dimensions]),
    Dense(units=128, activation='relu'),
    Dropout(0.15),
    BatchNormalization(),
    Dense(units=128, activation='relu'),
    Dropout(0.15),
    BatchNormalization(),
    Dense(units=number_classes, activation='softmax')
])

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy'
)

history = model.fit(X_train, y_train_cat,
          validation_data=(X_valid, y_valid_cat),
          batch_size=32,
          epochs=200,
          callbacks=[early_stopping],
          verbose=0 # turn off training log
                    )
```

## Quantitative Analysis
- Models were assessed primarily with **Categorical Cross Entropy** and **Accuracy**. See details below:
  
| Train Categorical Cross Entropy    | Validation Categorical Cross Entropy   | Test Categorical Cross Entropy   |
|-------------|-------------|-------------|
| 0.0567 | 0.0908 | Unknown |

| Train Accuracy    | Validation Accuracy   | Test Accuracy   |
|-------------|-------------|-------------|
| 0.9954 | 0.9748 | Unknown |

## Loss Curve
![Loss Curve](images/Loss%20Curve.png)

## Ethical Considerations
- **Potential Negative impacts of using model**:
  - **Math and software related problems**:
    - Model took more than 6 minutes to train on Windows laptop. Lengthy process of fine-tuning parameters, e.g., batch size, the patience, and number of nodes within each hidden layer, for my model to eventually fit and converge with desired validation loss.
    - Given the number of inputs applied to the model and the large number of nodes designated within each hidden layer, as well as a Dropout function applied to introduce random shutting off of neurons during training, tracing the decision making process of the algorithm would be practically impossible.
    - The high number of hidden nodes applied can lead to potential overfitting.
    - Model may have issues in predicting digits that are underrepresented in the dataset. In this case, may be appropriate to add class weights to the loss function.
    - Training is slow. A fast CPU and GPU likely necessary to obtain better loss and accuracy metrics.
  - ** Real-world risks: Who, What, When or How**:
    - 










