# Deep Learning Challenge: Alphabet Soup Charity Funding Success Prediction

## Overview of the Analysis

The purpose of this analysis is to develop a binary classification model that predicts whether an organization funded by Alphabet Soup will be successful in its ventures. The model uses features such as application type, organization classification, income, funding amount requested, and others, in order to determine the likelihood of success (`IS_SUCCESSFUL`). The goal is to utilize machine learning techniques to optimize and evaluate the model's performance.

## Results

### Data Preprocessing

- **Target Variable(s):**
  - The target variable for this model is `IS_SUCCESSFUL`, which indicates whether the funding was used effectively (1 for success, 0 for failure).

- **Feature Variables:**
  - The features for this model are all columns except `EIN` and `NAME`, which were dropped due to being identification columns. The relevant features include:
    - APPLICATION_TYPE
    - AFFILIATION
    - CLASSIFICATION
    - USE_CASE
    - ORGANIZATION
    - STATUS
    - INCOME_AMT
    - SPECIAL_CONSIDERATIONS
    - ASK_AMT

- **Columns Removed:**
  - The `EIN` and `NAME` columns were removed because they do not contain useful information for predicting success.

- **Unique Values in Each Column:**
  - The number of unique values for each column was assessed, and for those with more than 10 unique values, the frequency of data points for each unique value was calculated. For categorical variables with rare occurrences, a new value called "Other" was created to group the less frequent categories.

- **Rare Category Grouping:**
  - Rare categories in certain columns were grouped under the value "Other" to reduce the complexity of the model and handle sparsely represented data.

- **Data Encoding:**
  - Categorical variables were encoded using `pd.get_dummies()` to convert them into numeric format suitable for model training.

- **Splitting Data:**
  - The preprocessed data was split into features (`X`) and the target (`y`). Then, the data was divided into training and testing datasets using `train_test_split`.

- **Scaling Data:**
  - The features were scaled using `StandardScaler()` to normalize the data, ensuring that all feature values are on a similar scale and improve model performance.

### Compiling, Training, and Evaluating the Model

- **Model Architecture:**
  - A neural network model was built using TensorFlow and Keras. The input layer contained the number of features corresponding to the number of input features (after encoding categorical variables).
  - The architecture included:
    - An input layer
    - Two hidden layers with `relu` activation functions
    - An output layer with a `sigmoid` activation function (for binary classification)

- **Neurons and Layers:**
  - The model's architecture was chosen based on the number of input features. Each hidden layer had 64 neurons initially, with a second layer also having 64 neurons. The output layer had one neuron, as we are performing binary classification.

- **Compiling the Model:**
  - The model was compiled using the `binary_crossentropy` loss function, the `adam` optimizer, and accuracy as the evaluation metric.

- **Training the Model:**
  - The model was trained for 100 epochs with a batch size of 32, using the training dataset. During training, callbacks were used to save the model's weights every 5 epochs.

- **Evaluation:**
  - After training, the model was evaluated using the test dataset to determine its loss and accuracy. The initial model achieved an accuracy of 73%, which was below the target of 75%.

### Model Optimization

- **Optimization Attempts:**
  - To optimize the model and increase accuracy, the following strategies were implemented:
    1. Increased the number of epochs from 100 to 150 to allow the model more time to converge.
    2. Added a third hidden layer with 64 neurons to increase the model's capacity to learn from the data.
    3. Changed the activation function of the second hidden layer to `tanh` to experiment with improving model performance.
    4. Added a batch normalization layer after each hidden layer to help stabilize the learning process.

- **Results After Optimization:**
  - After implementing these optimizations, the model achieved an accuracy of 77%, exceeding the target of 75%. The final model was saved and exported as `AlphabetSoupCharity_Optimization.h5`.

## Summary

The final model achieved an accuracy of 77%, which meets the target performance. The neural network with three hidden layers, batch normalization, and an increased number of epochs successfully improved the model's predictive power. Further optimization techniques, such as adjusting activation functions and experimenting with different layer configurations, can further refine the model.

## Recommendation for a Different Model

Although a neural network is effective, a Random Forest Classifier could also be a good choice for this classification problem. Random forests are ensemble models that are robust to overfitting and perform well with structured data like the one we have. They are also less sensitive to hyperparameter tuning, making them a good alternative.

## Conclusion

The deep learning model developed for predicting the success of Alphabet Soup-funded organizations was successful in achieving the target performance after optimization. The analysis and model can be further refined with additional tuning or by exploring alternative models like Random Forests. The final neural network model shows promising results for this binary classification task.
