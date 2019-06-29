import sys, time
sys.path.append('..')
from utils import *

import argparse
import mxnet as mx
from mxnet import nd, gpu, gluon, init, autograd
from mxnet.gluon import Block, nn
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
        self.model.initialize(init=init.Xavier(), force_reinit=True)
        self.model.hybridize()
        self.model(nd.random.uniform(shape=(args.batch_size,1,self.board_x,self.board_y)))

        self.v_loss = gluon.loss.L2Loss()
        self.pi_loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
        self.trainer = gluon.Trainer(self.model.collect_params(),'adam',{'learning_rate': args.lr})

    def predict(self,x):
        x = x.as_in_context(self.ctx)
        p,v = self.model(x)
        return p,v

class Residual(Block):
    def __init__(self, channels, kernel_size, padding=(1, 1), **kwargs):
        super(Residual, self).__init__(**kwargs)
        with self.name_scope():
            self.residual = nn.Sequential()
            self.residual.add(nn.Conv2D(channels=channels, kernel_size=kernel_size, padding=(1, 1)))
            self.residual.add(nn.BatchNorm(axis=3))
            self.residual.add(nn.Activation('relu'))
            self.residual.add(nn.Conv2D(channels=channels, kernel_size=kernel_size, padding=(1, 1)))
            self.residual.add(nn.BatchNorm(axis=3))

            self.relu = nn.Activation('relu')

    def forward(self, x):
        shortcut = x.copy()
        ff_result = self.residual(x)
        residual_result = self.relu(shortcut + ff_result)
        return residual_result

class TicTacToeResNNet():
    def __init__(self, game, args):
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.model = nn.Sequential()
        self.model.add(nn.Conv2D(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        self.model.add(nn.BatchNorm(axis=3))
        self.model.add(nn.Activation('relu'))

        self.model.add(Residual(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        self.model.add(Residual(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        self.model.add(Residual(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        self.model.add(Residual(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        self.model.add(Residual(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        self.model.add(Residual(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        self.model.add(Residual(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        self.model.add(Residual(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        self.model.add(Residual(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        self.model.add(Residual(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        self.model.add(Residual(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        self.model.add(Residual(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        self.model.add(Residual(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        self.model.add(Residual(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        self.model.add(Residual(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        self.model.add(Residual(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        self.model.add(Residual(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        self.model.add(Residual(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
        self.model.add(Residual(channels=args.num_channels, kernel_size=3, padding=(1, 1)))

        self.model.add(nn.Conv2D(channels=args.num_channels, kernel_size=3, padding=(1, 1)))
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

        self.model.initialize(init=init.Xavier(), force_reinit=True)
        self.model.hybridize()
        self.model(nd.random.uniform(shape=(args.batch_size,1,self.board_x,self.board_y)))


        self.v_loss = gluon.loss.L2Loss()
        self.pi_loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
        self.trainer = gluon.Trainer(self.model.collect_params(),'adam',{'learning_rate': args.lr})

    def predict(self,x):
        p,v = self.model(x)
        return p,v





