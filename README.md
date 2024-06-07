# Predicting Bank Marketing Campaign Success
### Project by:
* [Saudia Epps](https://github.com/smepps1989)
* [Clayton Knight](https://github.com/claytonmknight)
* [Erik Malrait](https://github.com/virtule)
* [Jacqueline Meyer](https://github.com/JacqueLeeMeyer)


# Introduction to the Dataset:
The Bank Marketing Dataset is derived from direct marketing campaigns (via phone calls) conducted by a Portuguese banking institution. The primary objective of this dataset is to facilitate the classification task of predicting whether a client will subscribe to a term deposit based on various attributes collected during these campaigns.

## Available Datasets:
* **bank-additional-full.csv**: Contains all examples (41,188) with 20 features, ordered by date from May 2008 to November 2010. 
* **bank-additional.csv**: A 10% sample of the first dataset (4,119 examples), randomly selected, with 20 features.
* **bank-full.csv**: An older version of the full dataset with all examples and 17 features, ordered by date.
* **bank.csv**: A 10% sample of the third dataset (4,521 examples), randomly selected, with 17 features.

We decided on using the **bank-full.csv**, as it contained the full dataset with fewer unnecessary features, saving us from having to drop those later on.

## Classification Goal:
The main use of this dataset is to predict whether a client will agree to subscribe to a [term deposit](https://www.investopedia.com/terms/t/termdeposit.asp) after being contacted through the bank's marketing campaign.

Our goal is to use the data collected from these phone calls such as; the client's age, job, marital status, and previous interactions with the bank — to build a model that can determine if the client will say 'yes' or 'no' to opening a term deposit.

# Creating the Model:
In our project to predict whether a client will subscribe to a term deposit, we explored several machine-learning approaches to find the most effective model. Below, we outline the different models we tried and the techniques we used to enhance their performance.

## Logistic Regression
Our first approach was to implement a Logistic Regression model.

**Implementation**: We used the Logistic Regression implementation from the **scikit-learn** library. <br>
**Results**: The model provided decent initial results, giving us a benchmark to improve upon: 

| Outcome  | Precision | Recall | F1 Score |
| -------- | --------- | ------ | -------- |
| no  `0`  | `92%`     | `97%`  | `95%`    |
| yes `1`  | `64%`     | `35%`  | `45%`    |

**Evaluation**: While the precision for the 'yes' class is relatively high `64%`, the recall is quite low `35%`. This indicates that while the model is good at predicting clients who will join (when it does predict this), it misses a rather large portion of actual clients.

## Random Forest
Our next approach was to use a Random Forest model.

**Implementation**: We used the Random Forest implementation from **scikit-learn**. <br>
**Results**: The model gave us very similar results to the Logistic Regression model, confirming we should try something else: 

| Outcome  | Precision | Recall | F1 Score |
| -------- | --------- | ------ | -------- |
| no  `0`  | `92%`     | `97%`  | `95%`    |
| yes `1`  | `65%`     | `39%`  | `49%`    |

**Evaluation**: With very similar results to the Logistic Regression model, we decided to use Neural Network to try and find a different result. <br>

## Neural Network
Given the results from Random Forest and Logistic Regression, we decided to explore Neural Networks for potentially better outcomes. 

**Implementation**: We used the **Keras** library with a TensorFlow backend to build and train our Neural Network models. <br>
**Results**: The model gave us different results compared to the previous models: 

| Trial # |  Testing Data   |  Testing Data   |   Training Data |   Training Data | Layers  | Neurons | Function |
| ------- | --------------- | --------------- | --------------- | --------------- | ------- | ------- | -------- |
|   `1`   | Accuracy        | Loss            | Accuracy        | Loss            | Layer 1 | `6`     | relu     |
|   `1`   | `90.3%`         | `21.3%`         | `91.0%`         | `19.4%`         | Layer 2 | `1`     | sigmoid  |

| Epochs |
| ------ |
| `100`  |

**Evaluation**: The Neural Network gave much better results, higher overall accuracy and somewhat low loss. We were satisfied with these as a base, so we moved on to begin optimizing<br>

# Optimizations:
To enhance the performance of our Neural Network, we undertook a series of optimization steps:

## Changes in Neurons and Layers
* **Neurons**: We experimented with different numbers of neurons in each layer to find the optimal configuration. Increasing the number of neurons generally allows the model to capture more complex patterns, but it also increases the risk of overfitting.
* **Layers**: We tested various architectures with 2, 3, and 4 layers to determine the best depth for our neural network:

## Activation Functions
We tested several activation functions to determine which worked best for our model:

* **ReLU (Rectified Linear Unit)**: Commonly used for hidden layers, ReLU helps mitigate the vanishing gradient problem.
* **Sigmoid**: Often used for the output layer in binary classification tasks.
* **Tanh**: Provides zero-centered outputs, which can make optimization easier compared to the sigmoid function.
* **Leaky_ReLU**: An extension of ReLU that allows a small, non-zero gradient when the unit is not active, which helps to prevent dead neurons during training.
* **Softmax**: Used in the output layer for multi-class classification problems to convert raw scores into probabilities, ensuring that the sum of the probabilities is 1.

## SMOTE / Synthetic Minority Over-sampling Technique ([Documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html))
At this point in the optimization stages, we took a look at the class imbalance in our dataset, we applied SMOTE. This technique generates synthetic samples for the minority class, thereby balancing the dataset and improving the model's ability to learn from the minority class

## Early Stopping Callback ([Documentation](https://keras.io/api/callbacks/early_stopping/))
To prevent overfitting and ensure the model generalizes well to unseen data, we implemented the Early Stopping callback:

* **Patience Parameter**: This parameter was tuned to stop training when the model's performance on the validation set did not improve for a specified number of epochs.
* **Monitoring**: We monitored the validation loss and stopped training once it ceased to decrease, thus preventing the model from overfitting to the training data.

# Optimization Results:
As we applied all these different optimizations we saw a variety of results and changes, so we created a chart to track every single test. This lets us quickly record the results of each trial, then quickly change and run the model again to be as efficient as possible.

Below is a table with all of the optimizations and the results of each change:
| Trial # |  Testing Data   |  Testing Data   |   Training Data |   Training Data | Layers  | Neurons | Function | Epochs | SMOTE      | Early Stop |
| ------- | --------------- | --------------- | --------------- | --------------- | ------- | ------- | -------- | ------ | ---------- | ---------- |
|   `2`   | Accuracy        | Loss            | Accuracy        | Loss            | Layer 1 | `30`    | relu     |  `100` | Not Active | Not Active |
|   `2`   | `90.1%`         | `22.7%`         | `92.5%`         | `17.2%`         | Layer 2 | `1`     | sigmoid  |  `100` | Not Active | Not Active |
|   `3`   | Accuracy        | Loss            | Accuracy        | Loss            | Layer 1 | `30`    | relu     |  `100` | `1`        | Not Active |
|   `3`   | `86.6%`         | `34.4%`         | `91.9%`         | `21.2%`         | Layer 2 | `1`     | sigmoid  |  `100` | `1`        | Not Active |
|   `4`   | Accuracy        | Loss            | Accuracy        | Loss            | Layer 1 | `30`    | relu     |  `100` | `0.5`      | Not Active |
|   `4`   | `88.0%`         | `29.0%`         | `90.9%`         | `22.4%`         | Layer 2 | `1`     | sigmoid  |  `100` | `0.5`      | Not Active |
|   `5`   | Accuracy        | Loss            | Accuracy        | Loss            | Layer 1 | `30`    | relu     |  `100` | `0.5`      | Not Active |
|   `5`   | `87.3%`         | `38.6%`         | `93.8%`         | `15.4%`         | Layer 2 | `30`    | tanh     |  `100` | `0.5`      | Not Active |
|   `5`   |                 |                 |                 |                 | Layer 3 | `1`     | sigmoid  |  `100` | `0.5`      | Not Active |
|   `6`   | Accuracy        | Loss            | Accuracy        | Loss            | Layer 1 | `30`    | relu     |  `100` | `0.5`      | Not Active |
|   `6`   | `11.0%`         | `15.0%`         | `33.0%`         | `39.0%`         | Layer 2 | `30`    | tanh     |  `100` | `0.5`      | Not Active |
|   `6`   |                 |                 |                 |                 | Layer 3 | `1`     | softmax  |  `100` | `0.5`      | Not Active |
|   `7`   | Accuracy        | Loss            | Accuracy        | Loss            | Layer 1 | `30`    | relu     |  `100` | `0.5`      | Not Active |
|   `7`   | `87.0%`         | `34.8%`         | `91.0%`         | `21.1%`         | Layer 2 | `30`    |leaky_relu|  `100` | `0.5`      | Not Active |
|   `7`   |                 |                 |                 |                 | Layer 3 | `1`     | sigmoid  |  `100` | `0.5`      | Not Active |
|   `8`   | Accuracy        | Loss            | Accuracy        | Loss            | Layer 1 | `30`    | relu     |  `100` | `0.5`      | `Stopped at 7` |
|   `8`   | `88.2%`         | `34.0%`         | `92.7%`         | `17.9%`         | Layer 2 | `30`    |leaky_relu|  `100` | `0.5`      | `Stopped at 7` |
|   `8`   |                 |                 |                 |                 | Layer 3 | `1`     | sigmoid  |  `100` | `0.5`      | `Stopped at 7` |
|   `9`   | Accuracy        | Loss            | Accuracy        | Loss            | Layer 1 | `30`    | relu     |  `100` | `0.5`      | `Stopped at 5` |
|   `9`   | `88.0%`         | `26.1%`         | `90.1%`         | `23.5%`         | Layer 2 | `30`    | relu     |  `100` | `0.5`      | `Stopped at 5` |
|   `9`   |                 |                 |                 |                 | Layer 3 | `1`     | sigmoid  |  `100` | `0.5`      | `Stopped at 5` |
|   `10`  | Accuracy        | Loss            | Accuracy        | Loss            | Layer 1 | `30`    | relu     |  `100` | `0.5`      | `Stopped at 6` |
|   `10`  | `87.7%`         | `25.8%`         | `91.8%`         | `19.8%`         | Layer 2 | `30`    | relu     |  `100` | `0.5`      | `Stopped at 6` |
|   `10`  |                 |                 |                 |                 | Layer 3 | `30`    | tanh     |  `100` | `0.5`      | `Stopped at 6` |
|   `10`  |                 |                 |                 |                 | Layer 4 | `1`     | sigmoid  |  `100` | `0.5`      | `Stopped at 6` |
|   `11`  | Accuracy        | Loss            | Accuracy        | Loss            | Layer 1 | `30`    | relu     |  `100` | `0.5`      | `Stopped at 4` |
|   `11`  | `88.1%`         | `26.0%`         | `87.5%`         | `27.3%`         | Layer 2 | `30`    | relu     |  `100` | `0.5`      | `Stopped at 4` |
|   `11`  |                 |                 |                 |                 | Layer 3 | `30`    | relu     |  `100` | `0.5`      | `Stopped at 4` |
|   `11`  |                 |                 |                 |                 | Layer 4 | `1`     | sigmoid  |  `100` | `0.5`      | `Stopped at 4` |
|   `12`  | Accuracy        | Loss            | Accuracy        | Loss            | Layer 1 | `144`   | relu     |  `100` | `0.5`      | `Stopped at 1` |
|   `12`  | `86.9%`         | `26.5%`         | `88.1%`         | `27.8%`         | Layer 2 | `96`    | relu     |  `100` | `0.5`      | `Stopped at 1` |
|   `12`  |                 |                 |                 |                 | Layer 3 | `48`    | relu     |  `100` | `0.5`      | `Stopped at 1` |
|   `12`  |                 |                 |                 |                 | Layer 4 | `1`     | sigmoid  |  `100` | `0.5`      | `Stopped at 1` |
