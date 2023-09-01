# TensorFlow Advanced Technic Specialization - Coursera

This repo contain all the materials from [Coursera](https://www.coursera.org/specializations/tensorflow-advanced-techniques).

## C1 :  Custom Models, Layers, and Loss function with TensorFlow

### Week 1 : Functional APIs

- [Functional API](./C1/week_1/C1_W1_Lab_1_functional-practice.ipynb)
- [Functional API usage](./C1/week_1/C1_W1_Lab_2_multi-output.ipynb)
- [Siamese Network](./C1/week_1/C1_W1_Lab_3_siamese-network.ipynb)
- [Assignment](./C1/week_1/C1W1_Assignment.ipynb)


### Week 2 : Custom Loss functions

- Custom Loss functions
- Custom Loss Class and hyperparameters
- Contrastive Loss
- Assignment



### Week 3 : Custom Layers

- Custom Lambda Layers
- Custom Lambda Layers : Usage
- Activation of custom layers
- Assignment


### Week 4 : Custom Models

- Complex architectures with the functional API
- Simplify complex architectures with class Model
- RestNet implementation
- Assignment


### Week 5 : Callbacks

- Integrated reminder
- Personnalized reminder

## C2 : Custom and distributed  Training with TensorFlow

### Week 1 : Gradients and differenciation 

- Tensor Notions
- Tensor in eaer mode
- Assingment

### Week 2 : Custom functions

- Custom training loops
- Custom pipeline with TensorFlow
- Assignment

### Week 3 : Graphic mode

- Autograph
- Complex code graph building
- Assignment

### Week 4 : Distributed Strategy

- Mirror strategy
- Mirror multi-GPU Strategy
- TPU strategy
- Assignment

## C3 : 

### Week 1 : Introduction to copmputer Vision

- Transfert learning
- Advanced transfert learning
- Detection and object localization
- Assignment

### Week 2 : Object detection

- Object detection with TensorFlow
- Object detection API
- Assignment

### Week 3 : Image segentation

- Image segmentation overview : FCNN
- U-Net
- Instance segmentation
- Assignment

# Snipets

## Creating models with Sequential

```python
tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                            tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                            tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
```

## Creting models with API

```python

# instantiate the input Tensor
from tensorflow.keras.models import Model

input_layer = tf.keras.Input(shape=(28, 28))

# stack the layers using the syntax: new_layer()(previous_layer)
flatten_layer = tf.keras.layers.Flatten()(input_layer)
first_dense = tf.keras.layers.Dense(128, activation=tf.nn.relu)(flatten_layer)
output_layer = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(first_dense)

# declare inputs and outputs
func_model = Model(inputs=input_layer, outputs=output_layer)
```


## Plot Model

```python
from tensorflow.python.keras.utils.vis_utils import plot_model

plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')
```

## Build model and train

```python
# configure, train, and evaluate the model
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
```
## Multiple Ouput Models

```python
#Define model layers.
input_layer = Input(shape=(len(train .columns),))
first_dense = Dense(units='128', activation='relu')(input_layer)
second_dense = Dense(units='128', activation='relu')(first_dense)

# Y1 output will be fed directly from the second dense
y1_output = Dense(units='1', name='y1_output')(second_dense)
third_dense = Dense(units='64', activation='relu')(second_dense)

# Y2 output will come via the third dense
y2_output = Dense(units='1', name='y2_output')(third_dense)

# Define the model with the input layer and a list of output layers
model = Model(inputs=input_layer, outputs=[y1_output, y2_output])

# Specify the optimizer, and compile the model with loss functions for both outputs
optimizer = tf.keras.optimizers.SGD(lr=0.001)
model.compile(optimizer=optimizer,
              loss={'y1_output': 'mse', 'y2_output': 'mse'},
              metrics={'y1_output': tf.keras.metrics.RootMeanSquaredError(),
                       'y2_output': tf.keras.metrics.RootMeanSquaredError()})
# Train the model for 500 epochs
history = model.fit(norm_train_X, train_Y,
                    epochs=500, batch_size=10, validation_data=(norm_test_X, test_Y))

# Test the model and print loss and mse for both outputs
loss, Y1_loss, Y2_loss, Y1_rmse, Y2_rmse = model.evaluate(x=norm_test_X, y=test_Y)
print("Loss = {}, Y1_loss = {}, Y1_mse = {}, Y2_loss = {}, Y2_mse = {}".format(loss, Y1_loss, Y1_rmse, Y2_loss, Y2_rmse))
```

## Multiple Input Model - Siamese NN

```python

```




