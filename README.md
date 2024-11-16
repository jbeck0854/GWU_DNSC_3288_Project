# GWU_DNSC_3288_Project
Digit Recognizer. Introduction to Deep Learning, Neural Networks, and Pattern Recognition.

## Basic Information
- **Person or organization developing model**: Jonathan, jon@yahoo.com
- **Model date**: November, 2024
- **Model version**: 1.0
- **License**: MIT *?*
- **Model implementation code**: https://colab.research.google.com/drive/1KXlQ71soA8vGusOqcWzRjidf3ikVfA7K?usp=drive_link

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
  - **Real-World risks: Who, What, When or How**:
    - **Who**: Banks, financial institutions, financial systems. **What**: Misclassified digits leading to incorrect decisions. **When**: Anytime a similar predictive digit rezognition algorithm automates decision making processes without oversight. **How**: Incorrect withdrawal amounts dispersed by ATMs or banking applications when a handwritten check is inputted; misclassification of handwritten checks leading to funds being deposited into incorrect account.
    - **Who**: Individuals who submit handwritten forms to be processed and stored. **What**: Privacy and processing concerns. **When**: Anytime a handwritten assessment or form is submitted to be processed automatically. **How**: Handwritten forms submitted at educational institutions, in which the documents are processed by computer systems, may incorrectly label and store the data; students may be unfailrly penalized when taking assessments unless they are graded manually; consensus data submitted by individuals by mail that are likely processed in an automatic fashion may be incorrectly stored, leading to faulty profiles.
    - **Who**: Any entity that uses a similar algorithm to automate high stakes processes. **What**: A poorly deisgned model that is hastily deployed, lacking proper governance, oversight, and controls, can negatively affect people at scale.**When**: When manual review is viewed by an entity as redundant, unneccesary, and costly to the organization **How**: Automated processing of forms and files by government, public, and health care services may lead to errors in records and billing.
  - **Potential uncertainties relating to the impacts of using model**:
    - **Math and Software Related Problems**:
      -  Vast number of computations taking place throughout the model at all times, would be extremely difficult to track manually.
      -  Training was stopped when 10 epochs did not lead to noticeable improvement in validation loss. This number was chosen because the model could no longer converge (potentially due to lack of CPU power) with a larger parameter chosen.
      -  Determining the 'best model' is largely a procedure of random guessing at the onset of building the architecture of model.
        - Black-box nature of model design. Very likely that additional parameters could have been used to improve loss and accuracy metrics.
      - Interpretation and explainability of why some parameter (e.g., choosing 'rmsprop' over the 'adam' SGD optimizer) works better on model is difficult.
      - 'ReLu' activation function can lead to 'dead neurons', causing fitting of model to slow drastically, if not end altogether, without notifying the programmer that this has occurred.
      - Training data may not accurately represent population (e.g., if training contains all neatly written digits and testing contains sloppily written digits, or vice-versa).
      - Differences in CPU/GPU quality may lead to different outcomes.
      - Small changes in inputs could lead to large changes in results of model.
      - Black-box nature causes difficulties with debugging.
    - **Real-world uncertainty risks: Who, What, When, and How**:
      - **Who**: Any entity that relies on model for critical decision making processes. **What**: Misclassification leading to inaccurate, unjust outcomes for individuals at scale. **When**: Anytime these models are deployed without proper governance, controls, oversight, and appeal processes in place; regulatory compliance measures are inconsistent; when security on similar models is breached - hard to pinpoint where exactly breached occured unless model properly document and monitored at all time, given so many parameters and inputs **How**: Model outputs invalid outputs for an individual and the individual can not fight the inaccuracies of model because the developers don't have proper understanding of its decision making processes; decision makers aren't able to efficiently interpret the final decision to an individual that the model affects; model was rushed to be deployed for first-mover advantage and so owners of model either aren't able to or are unwilling to attempt to explain its functions to end-users; most institutions have no incentive to make their models more transparent so interpretable and explainable outcomes likely unrealistic in most settings where similar models are deployed; if the model can't be explained, debugging the model and providing users the right to appeal its decision making processes, is not possible; end-users could manipulate the model to their benefit if they realize e.g., an Aversarial Examples attack.
  - **Description of unexpected results**:
    - 










