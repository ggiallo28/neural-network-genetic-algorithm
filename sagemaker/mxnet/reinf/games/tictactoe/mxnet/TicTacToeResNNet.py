import sys
sys.path.append('..')
from mxnet.gluon.model_zoo import vision
from mxnet.gluon import Block, nn, HybridBlock
from mxnet import init, nd, gluon
import mxnet as mx

class DDense(HybridBlock):
    def __init__(self, action_size, **kwargs):
        super(DDense, self).__init__(**kwargs)
        with self.name_scope():
            self.pi = nn.Dense(action_size, activation='softrelu')
            self.v = nn.Dense(1, activation='tanh')
            self.pi.initialize(init=init.Xavier(), force_reinit=True)
            self.v.initialize(init=init.Xavier(), force_reinit=True)

    def hybrid_forward(self, F, x):
        pi = self.pi(x)
        v = self.v(x)
        return [pi,v]

class TicTacToeNNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        if args.cuda:
            self.ctx = mx.gpu()
        else:
            self.ctx = mx.cpu()

        self.model = vision.squeezenet1_1(pretrained=True)
        hybrid_sequential = nn.HybridSequential()
        conv_layer = nn.Conv2D(channels=64, kernel_size=3, padding=(1, 1))
        conv_layer.initialize(init=init.Xavier(), force_reinit=True)
        hybrid_sequential.add(conv_layer)
        for layer in self.model.features[1:]:
            hybrid_sequential.add(layer)

        with self.model.name_scope():
            self.model.features = hybrid_sequential
            self.model.output = nn.HybridSequential()
            self.model.output.add(DDense(self.action_size))

        print(self.model)
        self.model.initialize(init=init.Xavier(), force_reinit=True)
        self.model.hybridize()
        self.model(nd.random.uniform(shape=(args.batch_size,1,self.board_x,self.board_y)))

        self.v_loss = gluon.loss.L2Loss()
        self.pi_loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
        self.trainer = gluon.Trainer(self.model.collect_params(),'adam',{'learning_rate': args.lr})

    def predict(self,x):
        #start = time.time()
        p,v = self.model(x)
        #end = time.time()
        #print("predict time:", end - start)
        return p,v
