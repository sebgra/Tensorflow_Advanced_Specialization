# TensorFlow Advanced Technic Specialization - Coursera

This repo contain all the materials from [Coursera](https://www.coursera.org/specializations/tensorflow-advanced-techniques).

## C1 :  Custom Models, Layers, and Loss function with TensorFlow

### Week 1 : Functional APIs

- [Functional API](./C1/week_1/C1_W1_Lab_1_functional-practice.ipynb)
- [Functional API usage](./C1/week_1/C1_W1_Lab_2_multi-output.ipynb)
- [Siamese Network](./C1/week_1/C1_W1_Lab_3_siamese-network.ipynb)
- [Assignment](./C1/week_1/C1W1_Assignment.ipynb)


### Week 2 : Custom Loss functions

- [Custom Loss functions](./C1/week_2/C1_W2_Lab_1_huber-loss.ipynb)
- [Custom Loss Class and hyperparameters](./C1/week_2/C1_W2_Lab_2_huber-object-loss.ipynb)
- [Contrastive Loss](./C1/week_2/C1_W1_Lab_3_siamese-network.ipynb)
- [Assignment](./C1/week_2/C1W2_Assignment.ipynb)



### Week 3 : Custom Layers

- [Custom Lambda Layers](./C1/week_3/C1_W3_Lab_1_lambda-layer.ipynb)
- [Custom Lambda Layers : Usage](./C1/week_3/C1_W3_Lab_2_custom-dense-layer.ipynb)
- [Activation of custom layers](./C1/week_3/C1_W3_Lab_3_custom-layer-activation.ipynb)
- [Assignment](./C1/week_3/C1W3_Assignment.ipynb)


### Week 4 : Custom Models

- [Complex architectures with the functional API](./C1/week_4/C1_W4_Lab_1_basic-model.ipynb )
- Simplify complex architectures with class Model
- [RestNet implementation](./C1/week_4/C1_W4_Lab_2_resnet-example.ipynb)
- [Assignment](./C1/week_4/C1W4_Assignment.ipynb)


### Week 5 : Callbacks

- [Integrated reminder](./C1/week_5/C1_W5_Lab_1_exploring-callbacks.ipynb)
- [Personnalized reminder](./C1/week_5/C1_W5_Lab_2_custom-callbacks.ipynb)

## C2 : Custom and distributed  Training with TensorFlow

### Week 1 : Gradients and differenciation 

- [Tensor Notions](./C2/week_1/C2_W1_Lab_1_basic-tensors.ipynb)
- [Tensor in eaer mode](./C2/week_1/C2_W1_Lab_2_gradient-tape-basics.ipynb)
- [Assingment](./C2/week_1/C2W1_Assignment.ipynb)

### Week 2 : Custom functions

- [Custom training loops](./C2/week_2/C2_W2_Lab_1_training-basics.ipynb)
- [Custom pipeline with TensorFlow](./C2/week_2/C2_W2_Lab_2_training-categorical.ipynb)
- [Assignment](./C2/week_2/C2W2_Assignment.ipynb)

### Week 3 : Graphic mode

- [Autograph](./C2/week_3/C2_W3_Lab_1_autograph-basics.ipynb)
- [Complex code graph building](./C2/week_3/C2_W3_Lab_2-graphs-for-complex-code.ipynb)
- [Assignment](./C2/week_3/C2W3_Assignment.ipynb)

### Week 4 : Distributed Strategy

- [Mirror strategy](./C2/week_4/C2_W4_Lab_1_basic-mirrored-strategy.ipynb)
- [Mirror multi-GPU Strategy](./C2/week_4/C2_W4_Lab_2_multi-GPU-mirrored-strategy.ipynb)
- [TPU strategy](./C2/week_4/C2_W4_Lab_3_using-TPU-strategy.ipynb)()
- [TPU strategy 2](./C2/week_4/C2_W4_Lab_4_one-device-strategy.ipynb)()
- [Assignment](./C2/week_4/C2W4_Assignment.ipynb)

## C3 : Advanced computer vision ith TensorFlow

### Week 1 : Introduction to computer Vision

- [Transfert learning](./C3/week_1/C3_W1_Lab_1_transfer_learning_cats_dogs.ipynb)
- [Advanced transfert learning](./C3/week_1/C3_W1_Lab_2_Transfer_Learning_CIFAR_10.ipynb)
- [Detection and object localization](./C3/week_1/C3_W1_Lab_3_Object_Localization.ipynb)
- [Assignment](./C3/week_1/C3W1_Assignment.ipynb)

### Week 2 : Object detection

- [Object detection with TensorFlow](./C3/week_2/C3_W2_Lab_1_Simple_Object_Detection.ipynb)
- [Object detection API](./C3/week_2/C3_W2_Lab_2_Object_Detection.ipynb)
- [Assignment](./C3/week_2/C3W2_Assignment.ipynb)

### Week 3 : Image segentation

- [Image segmentation overview : FCNN](./C3/week_3/C3_W3_Lab_1_VGG16-FCN8-CamVid.ipynb)
- [U-Net](./C3/week_3/C3_W3_Lab_2_OxfordPets_UNet.ipynb)
- [Instance segmentation](./C3/week_3/C3_W3_Lab_3_Mask_RCNN_ImageSegmentation.ipynb)
- [Assignment](./C3/week_3/C3W3_Assignment.ipynb)

### Week 4 : Models visualization and interpretability

- [Introduction to visualization and interpretability](./C3/week_4/C3_W4_Lab_1_FashionMNIST_CAM.ipynb)
- [Introduction to visualization and interpretability 2](./C3/week_4/C3_W4_Lab_2_CatsDogs_CAM.ipynb)
- [Saliency maps](./C3/week_4/C3_W4_Lab_3_Saliency.ipynb)
- [Gradients and Class Activation Maps](./C3/week_4/C3_W4_Lab_4_GradCam.ipynb)
- [Assignment](./C3/week_4/C3W4_Assignment.ipynb)



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
# create the left input and point to the base network
input_a = Input(shape=(28,28,), name="left_input")
vect_output_a = base_network(input_a)

# create the right input and point to the base network
input_b = Input(shape=(28,28,), name="right_input")
vect_output_b = base_network(input_b)

# measure the similarity of the two vector outputs
output = Lambda(euclidean_distance, name="output_layer", output_shape=eucl_dist_output_shape)([vect_output_a, vect_output_b])

# specify the inputs and output of the model
model = Model([input_a, input_b], output)

# plot model graph
plot_model(model, show_shapes=True, show_layer_names=True, to_file='outer-model.png')
```

## Custom Loss function

```python

```

## Custom loss function with hyperparameter

```python

```





