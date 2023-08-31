# TensorFlow Advanced Technic Specialization - Coursera

This repo contain all the materials from [Coursera](https://www.coursera.org/specializations/tensorflow-advanced-techniques).

## C1 :  Custom Models, Layers, and Loss function with TensorFlow

### Week 1 : Functional APIs

- Functional API
- Functional API usage
- Siamese Network
- Assignment


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



