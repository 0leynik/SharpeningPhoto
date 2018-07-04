# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import Conv2D, Input
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
import h5py

def create_model():
    inputs = Input(shape=(None, None, 1))
    conv = Conv2D(128, (9, 9), padding='same', activation='relu')(inputs)
    conv = Conv2D(64, (3, 3), padding='same', activation='relu')(conv)
    outputs = Conv2D(1, (5, 5), padding='same', activation='linear')(conv)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(lr=0.0003), loss='mean_squared_error')

    return model


loss_save_dir = 'weights_loss'
if not os.path.isdir(loss_save_dir):
    os.makedirs(loss_save_dir)


def train():

    model = create_model()
    model.summary()

    f_train = h5py.File('../dataset/train.h5', 'r')
    f_val = h5py.File('../dataset/val.h5', 'r')

    data_train, label_train = f_train['data'], f_train['label']
    data_val, label_val = f_val['data'], f_val['label']

    train_on_batch = True

    if train_on_batch:
        f = file(os.path.join(loss_save_dir, 'metrics.csv'), 'w')

        iter_num = 0
        epoches = 10
        batch_size = 128
        save_model_step = 1000

        ids_train = np.arange(0, len(data_train))

        for e in range(1, epoches+1):
            print('epoch ' + str(e))
            np.random.shuffle(ids_train)
            ids_train_batchs = np.array_split(ids_train, int((len(data_train)/batch_size) + 1))

            for ids_tr_batch in ids_train_batchs:

                ids_tr_batch = sorted(ids_tr_batch)
                loss_train_batch = model.train_on_batch(data_train[ids_tr_batch], label_train[ids_tr_batch])

                ids_val_batch = sorted(np.random.choice(len(data_val), batch_size))
                loss_val_batch = model.test_on_batch(data_val[ids_val_batch], label_val[ids_val_batch])

                iter_num += 1

                print(str(datetime.now())+' iter ' + str(iter_num) + ', loss: ' + str(loss_train_batch) + ', val_loss: ' + str(loss_val_batch))

                metrics = str(iter_num) + ',' + str(loss_train_batch) + ',' + str(loss_val_batch)
                f.write(metrics)
                f.write('\n')
                f.flush()

                if (iter_num % save_model_step) == 0:
                    model.save(os.path.join(loss_save_dir, 'iter_' + str(iter_num) + '.h5'))

        f.close()
    else:
        loss_val_filepath = os.path.join(loss_save_dir, 'val_loss_e_{epoch:02d}_loss_{val_loss:.8f}.h5')
        loss_val_checkpoint = ModelCheckpoint(loss_val_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        history = model.fit(data_train[:], label_train[:], batch_size=128, validation_data=(data_val[:], label_val[:]), callbacks=[loss_val_checkpoint], shuffle=True, nb_epoch=200, verbose=1)
        # srcnn_model.load_weights("m_model_adam.h5")

        print(history.history.keys())

        f = file(os.path.join(loss_save_dir, 'metrics.csv'), 'w')
        loss = history.history['loss']
        loss_val = history.history['val_loss']
        for i in range(len(loss)):
            f.write(str(i+1)+','+str(loss[i])+','+str(loss_val[i]))
            f.write('\n')
            f.flush()
        f.close()

    f_train.close()
    f_val.close()


def plot_graph():

    metrics = np.loadtxt(os.path.join(loss_save_dir, 'metrics.csv'), delimiter=',')

    mpl.rcParams['figure.figsize'] = [8.4, 4.8]
    mpl.rcParams['figure.dpi'] = 500
    mpl.rcParams['lines.linewidth'] = 0.7
    mpl.rcParams['axes.linewidth'] = 0.3

    # plot graph loss
    plt.figure('Graph loss')
    plt.title('Loss')
    plt.plot(metrics[:, 1])
    plt.plot(metrics[:, 2])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    plt.grid(True, linestyle='--')
    # plt.yticks(np.linspace(0., 0.2, 11))
    # plt.ylim(0., 0.2)
    # plt.xticks(np.linspace(0, 5000, 11))
    # plt.xlim(0, 5000)
    plt.savefig(os.path.join(loss_save_dir, 'graph_history_loss.png'))

if __name__ == "__main__":
    train()
    # plot_graph()
