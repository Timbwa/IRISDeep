import tensorflow as tf
SEED = 1


def create_model(init_num_kernels=4, init_kernel_size=3, num_conv_layers=2, init_num_neurons_fc_layer=512,
                 num_of_fc_layers=2, strides=1, model_name='base_model', do_padding=True):
    model = tf.keras.models.Sequential(name=model_name)

    # add input layer (128x128) grayscale images
    model.add(tf.keras.layers.InputLayer(input_shape=[128, 128, 1], name='input_layer'))

    """
    parameters for convolutional layer:
        kernel_size: decrease with increase of convolutional layers
        kernel_number: increase by powers of 2 as we add additional max pooling layers
        strides: 1, 2, 3 ..., if strides > 1, do zero padding
        activation: RELU as declared in project
    pooling layer will remain as follows:
        kernel_size of 2x2
        stride of 2
    """

    # weight initializer
    initializer = tf.keras.initializers.he_normal(seed=SEED)
    # add at-least 2 convolutional layers
    # 2 conv layers has stride of 1 so do zero-padding. padding='valid' means no zero-padding, 'same' means padding
    # with zeros where the output size is same as input
    num = 1
    for i in range(num_conv_layers):
        if do_padding:
            model.add(tf.keras.layers.Conv2D(filters=init_num_kernels, kernel_size=init_kernel_size, strides=strides,
                                             padding='same', activation='relu', kernel_initializer=initializer,
                                             name=f'conv_{num}'))
        else:
            # don't add zero padding
            model.add(tf.keras.layers.Conv2D(filters=init_num_kernels, kernel_size=init_kernel_size, strides=strides,
                                             padding='valid', activation='relu', kernel_initializer=initializer,
                                             name=f'conv_{num}'))

        # add Batch Normalization Layer
        model.add(tf.keras.layers.BatchNormalization())
        # add max-pooling layer. Kernel size of 2x2 and stride of 2
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, padding='same', name=f'maxpool_{num}'))
        # double number of kernels
        if init_num_kernels < 512:
            init_num_kernels *= 2
        # decrease the kernel size
        if init_kernel_size > 1:
            init_kernel_size -= 2
        # increase variable for naming
        num += 1

    # Flatten MaxPool output before connecting to Fully Connected layers
    model.add(tf.keras.layers.Flatten())
    num = 1
    # default is 2 fully connected layer
    for i in range(num_of_fc_layers):
        model.add(tf.keras.layers.Dense(units=init_num_neurons_fc_layer, activation='relu',
                                        kernel_initializer=initializer, name=f'fc_layer_{num}'))
        # add Batch Normalization Layer
        model.add(tf.keras.layers.BatchNormalization())
        # add dropout layer with probability of 50% as seen in Hands-On Machine Learning with SciKit-Learn, Tensorflow
        # and Keras pg 473
        model.add(tf.keras.layers.Dropout(rate=0.5, seed=SEED, name=f'dropout_layer_{num}'))
        # decrease number of neurons by a factor of 2 for each subsequent fully-connected layer
        if init_num_neurons_fc_layer > 2:
            init_num_neurons_fc_layer /= 2
        num += 1

    soft_initializer = tf.keras.initializers.glorot_normal(seed=SEED)
    # connect with softmax layer for classification. We have 400 subjects, therefore 400 neurons
    model.add(tf.keras.layers.Dense(units=400, activation='softmax', kernel_initializer=soft_initializer,
                                    name='output_layer'))

    # print the model summary
    print(model.summary())

    return model


def compile_model(model: tf.keras.Model, learning_rate):
    """
    compiles the model with the following hyper-parameters:
    optimizer: RMSprop Optimizer
    loss: Softmax(Cross-entropy) -> SparseCategoricalCrossentropy for multiclass labels
    :param model: created keras model
    :param learning_rate:
    :return:
    """
    # compile model, returns None
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=tf.keras.metrics.SparseCategoricalAccuracy())
