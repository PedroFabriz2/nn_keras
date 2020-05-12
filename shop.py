import pandas as pd
import numpy as np
from keras.models import Sequential 
from keras.layers.core import Dense

TARGET_VARIABLE =  "usera"
TRAIN_TEST_SPLIT = 0.5
HIDDEN_LAYER_SIZE = 30
raw_data = pd.read_csv("data.csv")

#target_variable = our labels
#train_teste_split is where we split the dataset
#hidden_layer_size = number of neurons in the hidden layer

divide = np.random.randn(len(raw_data)) < TRAIN_TEST_SPLIT
train_dataset = raw_data[divide]
test_dataset = raw_data[~divide]

print(train_dataset['usera'])

train_data = np.array(train_dataset.drop("usera", axis=1))
train_label = np.array(train_dataset[[TARGET_VARIABLE]])
test_data = np.array(test_dataset.drop([TARGET_VARIABLE], axis=1))
test_label = np.array(test_dataset[[TARGET_VARIABLE]])

neural_net = Sequential()
neural_net.add(Dense(HIDDEN_LAYER_SIZE, input_shape=(3,), activation="sigmoid"))
neural_net.add(Dense(1, activation="sigmoid"))
neural_net.compile(loss="mean_squared_error", optimizer="sgd", metrics=['accuracy'])
neural_net.fit(train_data, train_label, epochs=150, batch_size=2, verbose=1)

metrics = neural_net.evaluate(test_data, test_label, verbose=1) 
print("%s: %.2f%%" % (neural_net.metrics_names[1], metrics[1]*100))

new_data = np.array(pd.read_csv("new_data.csv")) 
results = neural_net.predict(new_data)
print(results)

