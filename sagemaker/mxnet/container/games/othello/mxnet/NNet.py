import sys, os, time
import argparse
import shutil
import random
import math
sys.path.append('../..')
from utils import *
from NeuralNet import NeuralNet

import mxnet as mx
from mxnet import nd, gpu, gluon, init, autograd
import argparse
from .OthelloNNet import OthelloNNet as onnet

class NNetWrapper(NeuralNet):
    def __init__(self, game, args):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.name = str(hex(id(self)))
        self.loss = 99999999999
        self.game = game
        if args.cuda:
            self.ctx = mx.gpu()
            mx.ctx.default(ctx)
        else:
            self.ctx = mx.cpu()

    def train(self, train_data):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """

        end = time.time()
        input_boards, target_pis, target_vs = list(zip(*train_data))
        dataset_train = gluon.data.dataset.ArrayDataset(input_boards, target_pis, target_vs)
        data_loader = gluon.data.DataLoader(dataset_train,batch_size=args.batch_size,shuffle=True,num_workers=4)

        for epoch in range(args.epochs):
            for input_board, target_pi, target_v in data_loader:
                input_board = input_board.as_in_context(ctx)
                target_pi = target_pi.as_in_context(ctx)
                target_v = target_v.as_in_context(ctx)
                with autograd.record():
                    pi, v = self.nnet.predict(input_board)
                    self.pi_loss = self.nnet.pi_loss(pi,target_pi)
                    self.v_loss = self.nnet.v_loss(v,target_v)
                    self.loss = self.pi_loss + self.v_loss
                self.loss.backward()
                self.nnet.trainer.step(args.epochs)

        v0 = len(train_data)
        v1 = round(time.time()-end,2)
        v2 = round(self.loss.mean().asscalar(),5)
        v3 = round(self.pi_loss.mean().asscalar(),5)
        v4 = round(self.v_loss.mean().asscalar(),5)
        print('Examples {} | Time Total: {}s | loss {} | pi_loss {} | v_loss {}'.format(v0,v1,v2,v3,v4))


    def predict(self, board, isMCTS=False):
        if isMCTS:
            p,v = self.nnet.predict(board)
            return p[0],v[0]
        return self.nnet.predict(board)

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_parameters(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        print(filepath)
        if not os.path.exists(filepath):
            raise("No model in path '{}'".format(filepath))
        self.nnet.model.load_parameters(filepath)

    def get_weights(self):
        return self.nnet.model.get_weights()

    def set_weights(self, weights):
        self.nnet.model.set_weights(weights)

    def get_loss(self):
        return self.loss
