import sys
sys.path.append('..')
from utils import *

import argparse
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import mxnet as mx
from mxnet import nd, gpu, gluon, init, autograd
from mxnet.gluon import Block,nn
from keras.utils import plot_model
from tictactoe.TicTacToeGame import TicTacToeGame as Game
"""
NeuralNet for the game of TicTacToe.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloNNet by SourKream and Surag Nair.
"""
class CNN2D(Block):
    def __init__(self, channels, kernel, stride=1, padding=0, **kwargs):
        super(CNN2D, self).__init__(**kwargs)
        with self.name_scope():
            self.block = nn.Sequential()
            self.block.add(nn.Conv2D(channels=channels, kernel_size=kernel, strides=1, padding=padding))
            self.block.add(nn.BatchNorm())
            self.block.add(nn.Activation(activation='relu'))
            self.block.initialize(init=init.Xavier(),force_reinit=True)

    def forward(self, x):
        return self.block(x)

class Residual(Block):
    def __init__(self, channels, kernel_size, initial_stride, chain_length=1, stride=1, padding=0, **kwargs):
        super(RES_CNN2D, self).__init__(**kwargs)
        with self.name_scope():
            num_rest = chain_length - 1
            self.init_cnn = CNN2D(channels, kernel_size, initial_stride, padding)
            self.rest_cnn = nn.Sequential()
            for i in range(num_rest):
                self.rest_cnn.add(CNN2D(channels, kernel_size, stride, padding))

            self.ramp = nn.Activation(activation='relu')

    def forward(self, x):
        y = x.copy() # make a copy of untouched input to send through chuncks
        y = self.init_cnn(y)
        y = self.rest_cnn(y)
        y += x
        y = self.ramp(y)
        return y

class DDense(Block):
    def __init__(self, action_size, **kwargs):
        super(DDense, self).__init__(**kwargs)
        with self.name_scope():
            self.pi = nn.Dense(action_size, activation='softrelu')
            self.v = nn.Dense(1, activation='tanh')

    def forward(self, x):
        pi = self.pi(x)
        v = self.v(x)
        return [pi,v]

class TicTacToeNNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        net = nn.Sequential()
        net.add(nn.Conv2D(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        net.add(nn.BatchNorm(axis=3))
        net.add(nn.Activation('relu'))

        net.add(nn.Conv2D(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        net.add(nn.BatchNorm(axis=3))
        net.add(nn.Activation('relu'))

        net.add(nn.Conv2D(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        net.add(nn.BatchNorm(axis=3))
        net.add(nn.Activation('relu'))

        net.add(nn.Conv2D(channels=args.num_channels, kernel_size=3))
        net.add(nn.BatchNorm(axis=3))
        net.add(nn.Activation('relu'))

        net.add(nn.Flatten())

        net.add(nn.Dense(1024))
        net.add(nn.BatchNorm(axis=1))
        net.add(nn.Activation('softrelu'))
        net.add(nn.Dropout(args.dropout))

        net.add(nn.Dense(512))
        net.add(nn.BatchNorm(axis=1))
        net.add(nn.Activation('tanh'))
        net.add(nn.Dropout(args.dropout))

        net.add(DDense(self.action_size))

        net.initialize(init=init.Xavier(),force_reinit=True)
        net.hybridize()

        #board = nd.array([[Game().getInitBoard()]])
        #net.summary(board)
        #print(net(board))

        #n.initialize(init=init.Xavier(),force_reinit=True)
        #game = Game()
        #board = game.getInitBoard()
        #board = nd.array([[board]])
        #n.summary(board)
        #print(n(board))



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

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))
        self.model.summary()

#
