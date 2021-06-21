import models as m
import os
import acquisition as aq
import numpy as np


def print_equal():
    return ' ==================================== '


def get_run_logdir(exp_name):
    root_log_dir = os.path.join(os.curdir, "training_logs", exp_name)
    import time
    run_id = time.strftime("run_%d_%m_%Y-%H_%M_%S")
    return os.path.join(root_log_dir, run_id)


def train(model: m.tf.keras.Model, epochs, batch_size, train_x, train_y, val_x, val_y, exp_name):
    run_logdir = get_run_logdir(exp_name)
    callbacks = [
        # save the model at the end of each epoch(save_best_only=False) or save the model with best performance on
        # validation set(save_best_only=True)
        m.tf.keras.callbacks.ModelCheckpoint('iris_deep_model.h5', save_best_only=True),
        # perform early stopping when there's no increase in performance on the validation set in (patience) epochs
        m.tf.keras.callbacks.EarlyStopping(patience=115, restore_best_weights=True),
        # tensorboard callback
        m.tf.keras.callbacks.TensorBoard(run_logdir)
    ]
    """
        To view the training curves through tensorboard run the following command on terminal:

        $ tensorboard --logdir=C:\\Users\\PC\\CompVision\\IRISDeep\\training_logs\\experiment_name --port=6006

        make sure to replace --logdir path with absolute windows path(with single '\') of training_logs after training 
        starts
    """
    # fit the model using num of epochs and batch_size
    model.fit(x=train_x, y=train_y, validation_data=(val_x, val_y), epochs=epochs,
              batch_size=batch_size, callbacks=callbacks, verbose=True)


def evaluate_model(model: m.tf.keras.Model, test_x, test_y, name):
    # evaluate the model
    print(f'{print_equal()} Evaluation of {name} {print_equal()}')
    score = model.evaluate(x=test_x, y=test_y)
    print(f'Accuracy: {score[1]}')
    print(f'{print_equal()}')


def do_experiment(data, name='conv_layers', conv_layers=3, kernel_size=5):
    train_x, train_y, val_x, val_y, test_x, test_y = data
    init_kernel_size = 5
    learning_rate = 1e-3
    epochs = 250
    batch_size = 32
    if name == 'conv_layers':
        # do experiment for increasing number of conv layers from base model with constant number of fully connected
        # layers(2) stride = 1 and padding true
        for num_conv_layers in range(3, 7):
            print(f'{print_equal()} Conv-Layers: {num_conv_layers} {print_equal()}')
            model = m.create_model(init_num_kernels=4, init_kernel_size=init_kernel_size,
                                   num_conv_layers=num_conv_layers, init_num_neurons_fc_layer=512, num_of_fc_layers=2,
                                   strides=1, do_padding=True)
            m.compile_model(model, learning_rate)
            # train
            train(model, epochs, batch_size, train_x, train_y, val_x, val_y, exp_name=f'conv_layers_{num_conv_layers}')
            # evaluate
            evaluate_model(model, test_x, test_y, f'conv_layers_{num_conv_layers}')
            # increase kernel_size
            init_kernel_size += 2

    elif name == 'fc_layers':
        # do experiment for increasing number of conv layers from base model with constant number of fully connected
        # layers(2) stride = 1 and padding true

        for fc_layer in range(3, 6):
            print(f'{print_equal()} FC-Layers: {fc_layer} {print_equal()}')
            model = m.create_model(init_num_kernels=4, init_kernel_size=kernel_size,
                                   num_conv_layers=conv_layers, init_num_neurons_fc_layer=512,
                                   num_of_fc_layers=fc_layer,
                                   strides=1, do_padding=True)
            m.compile_model(model, learning_rate)
            # train
            train(model, epochs, batch_size, train_x, train_y, val_x, val_y, exp_name=f'fc_layers_{fc_layer}')
            # evaluate
            evaluate_model(model, test_x, test_y, f'conv_layers_{conv_layers}_fc_layers_{fc_layer}')

    elif name == 'strides':
        # do experiment with increasing number of strides, [2 - 4]. Done only with the current best model -> 3 conv
        # layers and 2 fc layers with initial stride = 1 and padding=True

        for stride in range(1, 5):
            print(f'{print_equal()} FC-Layers: {stride} {print_equal()}')
            model = m.create_model(init_num_kernels=4, init_kernel_size=init_kernel_size,
                                   num_conv_layers=3, init_num_neurons_fc_layer=512,
                                   num_of_fc_layers=2,
                                   strides=stride, do_padding=True)
            m.compile_model(model, learning_rate)
            # train
            train(model, epochs, batch_size, train_x, train_y, val_x, val_y, exp_name=f'strides_{stride}')
            # evaluate
            evaluate_model(model, test_x, test_y, f'conv_layers_{3}_strides_{stride}')


def load_best_model():
    """
        Load best model from disk with the following configuration:
        convolutional_layers: 3
        kernel sizes for the layers in order: 5x5, 3x3, 1x1
        kernel number for the layers: 4, 8, 16
        no-zero padding
        batch_norm & max_pool_layer after each conv layer
    """
    os.chdir('best_model')
    model_name = 'iris_deep_model.h5'
    best_model = m.tf.keras.models.load_model(model_name)
    return best_model


def main():
    # get data from numpy arrays
    print('Reading data...')
    train_x = np.load('train_x.npy')
    train_y = np.load('train_y.npy')
    val_x = np.load('val_x.npy')
    val_y = np.load('val_y.npy')
    test_x = np.load('test_x.npy')
    test_y = np.load('test_y.npy')
    print('Done!')

    # params for experiments
    learning_rate = 1e-3
    epochs = 250  # max epochs, early stopping may cause training to stop earlier
    batch_size = 32

    # create base model with 2 Conv layers and 1 fully-connected layer -> accuracy 75%
    # base_model = m.create_model(init_num_kernels=4, init_kernel_size=3, num_conv_layers=2, init_num_neurons_fc_layer=512,
    # num_of_fc_layers=2, strides=1, do_padding=True)
    #
    # m.compile_model(base_model, learning_rate)

    # train base model
    # print(f'{print_equal()} Training {print_equal()}')
    # train(base_model, epochs, batch_size, train_x, train_y, val_x, val_y, exp_name='base_model')
    # #
    # # # evaluate model
    # evaluate_model(base_model, test_x, test_y)

    # see the effect of increasing layers
    # do_experiment([train_x, train_y, val_x, val_y, test_x, test_y], name='conv_layers')

    # see effect of increasing fc layers
    # do_experiment([train_x, train_y, val_x, val_y, test_x, test_y], name='fc_layers', conv_layers=3, kernel_size=5)
    # do_experiment([train_x, train_y, val_x, val_y, test_x, test_y], name='fc_layers', conv_layers=4, kernel_size=7)
    # do_experiment([train_x, train_y, val_x, val_y, test_x, test_y], name='fc_layers', conv_layers=5, kernel_size=9)
    # do_experiment([train_x, train_y, val_x, val_y, test_x, test_y], name='fc_layers', conv_layers=6, kernel_size=11)

    # see effect of increasing strides with current best config of 3 conv layers, init_strides = 1, fc_layers=2
    # increasing stride reduces the feature size, killing the accuracy
    # do_experiment([train_x, train_y, val_x, val_y, test_x, test_y], name='strides')

    # try best configuration with padding False,
    current_best_model = m.create_model(init_num_kernels=4, init_kernel_size=5, num_conv_layers=3,
                                        init_num_neurons_fc_layer=512,
                                        num_of_fc_layers=2, strides=1, do_padding=False)

    m.compile_model(current_best_model, learning_rate)
    print(f'{print_equal()} Training {print_equal()}')
    train(current_best_model, epochs, batch_size, train_x, train_y, val_x, val_y, exp_name='conv_3_fc_2_padd_false')
    evaluate_model(current_best_model, test_x, test_y, f'conv_layers_{3}_padd_false')

    # load and evaluate best model
    best_model = load_best_model()
    # evaluate model
    evaluate_model(best_model, test_x, test_y, 'best_model')


if __name__ == '__main__':
    main()
