#import sys
#sys.path.append('..')
#from mxnet.gluon.model_zoo import vision
#from mxnet.gluon import Block, nn, HybridBlock
#from mxnet import init, nd, gluon
#import mxnet as mx
#
#class DDense(HybridBlock):
#    def __init__(self, action_size, **kwargs):
#        super(DDense, self).__init__(**kwargs)
#        with self.name_scope():
#            self.pi = nn.Dense(action_size, activation='softrelu')
#            self.v = nn.Dense(1, activation='tanh')
#            self.pi.initialize(init=init.Xavier(), force_reinit=True)
#            self.v.initialize(init=init.Xavier(), force_reinit=True)
#
#    def hybrid_forward(self, F, x):
#        pi = self.pi(x)
#        v = self.v(x)
#        return [pi,v]
#
#
#class TicTacToeNNet():
#    def __init__(self, game, args):
#        # game params
#        self.board_x, self.board_y = game.getBoardSize()
#        self.action_size = game.getActionSize()
#        self.args = args
#
#        if args.cuda:
#            self.ctx = mx.gpu()
#        else:
#            self.ctx = mx.cpu()
#
#        resnet18_v2 = vision.resnet18_v2()
#        self.model = nn.HybridSequential()
#        self.model.add(nn.Conv2D(channels=3, kernel_size=1))
#        for layer in resnet18_v2.features[:9]:
#            self.model.add(layer)
#
#        self.model.add(nn.Flatten())
#        self.model.add(nn.Dense(1024))
#        self.model.add(nn.BatchNorm(axis=1))
#        self.model.add(nn.Activation('softrelu'))
#        self.model.add(nn.Dropout(args.dropout))
#
#        self.model.add(nn.Dense(512))
#        self.model.add(nn.BatchNorm(axis=1))
#        self.model.add(nn.Activation('tanh'))
#        self.model.add(nn.Dropout(args.dropout))
#
#        self.model.add(DDense(self.action_size))
#
#        self.model.initialize(init=init.Xavier(), force_reinit=True)
#        self.model.hybridize()
#        self.model(nd.random.uniform(shape=(args.batch_size,1,self.board_x,self.board_y)))
#
#        self.v_loss = gluon.loss.L2Loss()
#        self.pi_loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
#        self.trainer = gluon.Trainer(self.model.collect_params(),'adam',{'learning_rate': args.lr})
#
#    def predict(self,x):
#        p,v = self.model(x)
#        return p,v
#
