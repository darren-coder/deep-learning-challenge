# deep-learning-challenge

Build a model to try to predict the success of charity organizations based on the provided information about them.

## Step 1: - Preprocess the Data

After importing the necessary dependencies, the 'charity_data.csv' was read into a pandas dataframe from the provided URL. Non-beneficial columns were removed.  The 'APPLICATION_TYPE' and 'CLASSIFICATION' columns had more than 10 values so a cutoff point was made to reduce that number to below 10 to ensure better performance from the model. Then all of the categorical data was to converted to numeric form using pandas 'pd.get_dummies'. After that the preprocessed data was split into features and target arrays and then divided into a training and testing dataset. An instance of 'StandardScaler' was created and fit to the training data. Finally the data was scaled using the 'transform' function to prepare it to be used in the model.

## Step 2: - Compile, Train, and Evaluate the Model

A neural network model was defined. I used a 'Sequential' model with 'relu' activation on the first 2 layers and 'sigmoid' on the final layer. Then the model was compiled and trained. This resulted in 72.84% accuracy, not quite reaching our goal of 75%.

### First model summary

Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (Dense)                   │ (None, 128)            │         5,632 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 1)              │            65 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 13,953 (54.50 KB)
 Trainable params: 13,953 (54.50 KB)
 Non-trainable params: 0 (0.00 B)

### First model evaluation

268/268 - 1s - 2ms/step - accuracy: 0.7284 - loss: 0.5782
Loss: 0.5782458782196045, Accuracy: 0.728396475315094

## Step 3: Optimize the Model

Since the results were underwhelming, some steps were taken to attempt optimization.

### Optimization 1

First I tried removing another column. In the 'SPECIAL_CONSIDERATIONS' column only 27 out of 34299 applications were answered 'N'. With 99.9% percent of the applications answer being 'Y' I thought this column might not be important for the outcome. After following all of the same steps as the first model, the result was 72.97% accuracy. Slightly better, but not enough to meet our goal.

#### Optimization 1 Structure

Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (Dense)                   │ (None, 128)            │         5,376 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 1)              │            65 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 13,697 (53.50 KB)
 Trainable params: 13,697 (53.50 KB)
 Non-trainable params: 0 (0.00 B)

#### Optimization 1 Evaluation

268/268 - 1s - 2ms/step - accuracy: 0.7297 - loss: 0.5715
Loss: 0.5715045928955078, Accuracy: 0.72967928647995

### Optimization 2

In this attempt I kept the 'SPECIAL_CONSIDERATIONS' column out since there was a slight improvement and also added another hidden layer with 'relu' activation. The results were 72.77% accuracy, noting a slight decrease from the previous attempt.

#### Optimization 2 Structure

Model: "sequential_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_3 (Dense)                 │ (None, 256)            │        10,752 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_4 (Dense)                 │ (None, 128)            │        32,896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_5 (Dense)                 │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_6 (Dense)                 │ (None, 1)              │            65 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 51,969 (203.00 KB)
 Trainable params: 51,969 (203.00 KB)
 Non-trainable params: 0 (0.00 B)

#### Optimization 2 Evaluation

268/268 - 1s - 2ms/step - accuracy: 0.7277 - loss: 0.6671
Loss: 0.6670857071876526, Accuracy: 0.7276967763900757

### Optimization 3

Since the first accuracy decreased a little bit with the extra layer, I removed it. There were 53.2% of the applications that were succesful as opposed to 46.8% unsuccessful. I thought that perhaps balancing things out with the 'compute_class_weight' function could help.





