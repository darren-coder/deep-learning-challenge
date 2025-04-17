# deep-learning-challenge

Build a model to try to predict the success of charity organizations based on the provided information about them.

## Step 1: - Preprocess the Data

After importing the necessary dependencies, the 'charity_data.csv' was read into a pandas dataframe from the provided URL. Non-beneficial columns were removed.  The 'APPLICATION_TYPE' and 'CLASSIFICATION' columns had more than 10 values so a cutoff point was made to reduce that number to below 10 to ensure better performance from the model. Then all of the categorical data was to converted to numeric form using pandas 'pd.get_dummies'. After that the preprocessed data was split into features and target arrays and then divided into a training and testing dataset. An instance of 'StandardScaler' was created and fit to the training data. Finally the data was scaled using the 'transform' function to prepare it to be used in the model.

## Step 2: - Compile, Train, and Evaluate the Model

A neural network model was defined. I used a 'Sequential' model with 'relu' activation on the first 2 layers and 'sigmoid' on the final layer. Then the model was compiled and trained. This resulted in 72.79% accuracy, not quite reaching our goal of 75%.

#### First model summary

Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (Dense)                   │ (None, 30)             │         1,320 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 15)             │           465 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 1)              │            16 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 1,801 (7.04 KB)
 Trainable params: 1,801 (7.04 KB)
 Non-trainable params: 0 (0.00 B)

#### First model evaluation

268/268 - 1s - 2ms/step - accuracy: 0.7279 - loss: 0.5593
Loss: 0.5592955350875854, Accuracy: 0.7279300093650818

## Step 3: Optimize the Model

Since the goal was not reached, some steps were taken to attempt optimization.

### Optimization 1

In the first optimization, I added another layer with 'relu' activation. The accuracy was slightly less at 72.75%.

#### Optimization 1 Structure

Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (Dense)                   │ (None, 45)             │         1,980 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 30)             │         1,380 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 15)             │           465 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 1)              │            16 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 3,841 (15.00 KB)
 Trainable params: 3,841 (15.00 KB)
 Non-trainable params: 0 (0.00 B)

#### Optimization 1 Evaluation

268/268 - 0s - 2ms/step - accuracy: 0.7275 - loss: 0.5596
Loss: 0.5595679879188538, Accuracy: 0.7274635434150696

### Optimization 2

In the second attempt I changed to 'tanh' activation and the results were at 72.89% accuracy. I also took the layer I added in optimization 1 away. The results were almost the same as the previous attempts, still not at the goal of 75%.

#### Optimization 2 Structure

Model: "sequential_3"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_11 (Dense)                │ (None, 30)             │         1,320 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_12 (Dense)                │ (None, 15)             │           465 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_13 (Dense)                │ (None, 1)              │            16 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 1,801 (7.04 KB)
 Trainable params: 1,801 (7.04 KB)
 Non-trainable params: 0 (0.00 B)

#### Optimization 2 Evaluation

268/268 - 0s - 2ms/step - accuracy: 0.7289 - loss: 0.5558
Loss: 0.5558069944381714, Accuracy: 0.728863000869751

### Optimization 3

In the third attempt, I went back to 'relu' activation. I used four layers total, three with 'relu' and the final layer with 'sigmoid'. 




