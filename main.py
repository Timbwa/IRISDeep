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
        # m.tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
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


def evaluate_model(model: m.tf.keras.Model, test_x, test_y):
    # evaluate the model
    print(f'{print_equal()}Evaluation{print_equal()}')
    score = model.evaluate(x=test_x, y=test_y)
    print(f'Accuracy: {score[1]}')
    print(f'{print_equal()}')


def main():
    # to do acquisition uncomment line below
    # train_x, train_y, val_x, val_y, test_x, test_y = aq.data_acquisition()

    # get data from numpy arrays from prior pre-processing
    train_x = np.load('train_x.npy')
    train_y = np.load('train_y.npy')
    val_x = np.load('val_x.npy')
    val_y = np.load('val_y.npy')
    test_x = np.load('test_x.npy')
    test_y = np.load('test_y.npy')

    # create base model with 2 Conv layers and 1 fully-connected layer
    base_model = m.create_model(init_num_kernels=4, init_kernel_size=3, num_conv_layers=2, init_num_neurons_fc_layer=512,
                                num_of_fc_layers=2, strides=1, do_padding=True)
    learning_rate = 1e-3
    m.compile_model(base_model, learning_rate)

    # train model
    epochs = 150  # max epochs, early stopping may cause training to stop earlier
    batch_size = 32

    print(f'{print_equal()} Training {print_equal()}')
    # train(base_model, epochs, batch_size, train_x, train_y, val_x, val_y, exp_name='base_model')
    #
    # # evaluate model
    evaluate_model(base_model, test_x, test_y)


if __name__ == '__main__':
    main()
