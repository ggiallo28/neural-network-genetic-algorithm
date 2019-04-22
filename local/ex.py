import tensorflow as tf
from keras import backend as K
from keras.models import load_model
import threading as t
import numpy as np
from tictactoe.TicTacToeGame import TicTacToeGame as Game
from utils import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *


K.clear_session()

args = dotdict({
    'numIters': 1,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 1,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

class CNN:
    def __init__(self):

        game = Game()

        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y

        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)                # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(h_conv2)))        # batch_size  x (board_x) x (board_y) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid')(h_conv3)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4_flat = Flatten()(h_conv4)
        s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))          # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1

        self.cnn_model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.cnn_model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))

        self.session = K.get_session()
        self.graph = tf.get_default_graph()
        with self.session.as_default():
            with self.graph.as_default():
                self.cnn_model.predict(np.empty((1, 3,3))) # warmup
        self.graph.finalize()

    def preproccesing(self, data):
        # dummy
        return data

    def query_cnn(self, data):
        X = self.preproccesing(data)
        with self.session.as_default():
            with self.graph.as_default():
                prediction = self.cnn_model.predict(X)
        print(prediction)
        return prediction


cnn = CNN()

th = t.Thread(target=cnn.query_cnn, kwargs={"data": np.empty((1, 3,3))})
th2 = t.Thread(target=cnn.query_cnn, kwargs={"data": np.empty((1, 3,3))})
th3 = t.Thread(target=cnn.query_cnn, kwargs={"data": np.empty((1, 3,3))})
th4 = t.Thread(target=cnn.query_cnn, kwargs={"data": np.empty((1, 3,3))})
th5 = t.Thread(target=cnn.query_cnn, kwargs={"data": np.empty((1, 3,3))})
th.start()
th2.start()
th3.start()
th4.start()
th5.start()

th2.join()
th.join()
th3.join()
th5.join()
th4.join()
