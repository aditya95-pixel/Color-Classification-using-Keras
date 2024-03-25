# Color Classification using Artificial Neural Networks
This project classifies colors into various categories using an artificial neural network implemented using TensorFlow/Keras. The color categories considered are Red, Green, Blue, Yellow, Orange, Pink, Purple, Brown, Grey, Black, and White.
## Dataset
The dataset used for training and testing is stored in a CSV file named final_data.csv. This dataset contains color samples along with their corresponding labels.
## Preprocessing
One-Hot Encoding: The color labels are one-hot encoded to convert categorical labels into a numerical format suitable for neural network training.

Splitting Dataset: The dataset is split into training and testing sets. The training set comprises 80% of the total dataset, while the remaining 20% is used for testing.
## Model Architecture
The neural network architecture consists of multiple dense layers:

Input Layer: Dense layer with ReLU activation, receiving input features.

Hidden Layers: Three hidden layers with ReLU activation functions, facilitating feature extraction.

Output Layer: Dense layer with 11 units, representing the probability distribution over color categories.

Regularization techniques, specifically L2 regularization with a coefficient of 0.001, are applied to prevent overfitting.
## Training
The model is compiled using Categorical Crossentropy loss and the Adam optimizer with a learning rate of 0.001. Training is carried out for 5001 epochs with a batch size of 2048. The training progress is monitored using validation data and the "EpochDots" callback.
## Evaluation
After training, the model is evaluated using the test dataset. Metrics such as accuracy and loss are computed to assess the model's performance. Additionally, a confusion matrix is generated to analyze the model's predictions.
## Result
The trained model achieves a satisfactory level of accuracy on the test dataset. The confusion matrix visualizes the model's performance across different color categories.
## Model Persistence
Finally, the trained model is saved to a file named "colormodel_trained_90.h5" for future use.
## Prediction
main1.py contains predict_color which uses the above trained model for predicting the color.
### web application built using streamlit
