import sys
sys.path.append('..')
from utils import *

import argparse
import mxnet as mx
from mxnet import nd, gpu, gluon, init, autograd
from mxnet.gluon import Block, nn
import numpy as np
"""
NeuralNet for the game of TicTacToe.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloNNet by SourKream and Surag Nair.
"""

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
        self.model = nn.Sequential()
        self.model.add(nn.Conv2D(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        self.model.add(nn.BatchNorm(axis=3))
        self.model.add(nn.Activation('relu'))

        self.model.add(nn.Conv2D(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        self.model.add(nn.BatchNorm(axis=3))
        self.model.add(nn.Activation('relu'))

        self.model.add(nn.Conv2D(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        self.model.add(nn.BatchNorm(axis=3))
        self.model.add(nn.Activation('relu'))

        self.model.add(nn.Conv2D(channels=args.num_channels, kernel_size=3))
        self.model.add(nn.BatchNorm(axis=3))
        self.model.add(nn.Activation('relu'))

        self.model.add(nn.Flatten())

        self.model.add(nn.Dense(1024))
        self.model.add(nn.BatchNorm(axis=1))
        self.model.add(nn.Activation('softrelu'))
        self.model.add(nn.Dropout(args.dropout))

        self.model.add(nn.Dense(512))
        self.model.add(nn.BatchNorm(axis=1))
        self.model.add(nn.Activation('tanh'))
        self.model.add(nn.Dropout(args.dropout))

        self.model.add(DDense(self.action_size))
        self.model.hybridize()
        self.model.initialize(init=init.Xavier(), force_reinit=True)

        self.v_loss = gluon.loss.L2Loss()
        self.pi_loss = gluon.loss.SoftmaxCrossEntropyLoss()
        self.trainer = gluon.Trainer(self.model.collect_params(),'adam')

    def predict(self,x):
        pi, v = self.model(nd.array(x))
        return pi.asnumpy(), v.asnumpy()

