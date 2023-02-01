import argparse, time, os, cfg, log
import numpy as np
from torch.autograd import Variable

from datasets.dataset import CrackData

from model import *

#########################################################################################
def adjust_learning_rate(optimizer, steps, step_size, gamma=0.1, logger=None):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * gamma
        if logger:
            logger.info('%s: %s' % (param_group['name'], param_group['lr']))

#########################################################################################
def cross_entropy_loss2d(inputs, targets, cuda=True, balance=1.1):  # cuda=False  lk
    """
    :param inputs: inputs is a 4 dimensional data nx1xhxw
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    n, c, h, w = inputs.size()
    # print('the value of n is %d'%n)  ### n=1
    weights = np.zeros((n, c, h, w))

    for i in range(n):      # xrange, lk
        t = targets[i, :, :, :].cpu().data.numpy()
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = neg + pos
        weights[i, t == 1] = neg * 1. / valid
        weights[i, t == 0] = pos * balance / valid

    weights = torch.Tensor(weights)

    if cuda:
        weights = weights.cuda()
    # inputs = F.sigmoid(inputs)
    # loss = nn.BCELoss(weights, size_average=False)(inputs, targets)

    loss = nn.BCEWithLogitsLoss(weights, size_average=False)(inputs, targets)

    return loss

def jacc_coef(prediction, label):

    smooth = 1.
    label_f = torch.flatten(label)
    prediction_f = torch.flatten(prediction)

    intersection = torch.sum(prediction_f * label_f)

    jacc = (2. * intersection + smooth) / (torch.sum(prediction_f) + torch.sum(label_f) - intersection + smooth)

    return jacc

def wcet_jacc(inputs, targets, cuda=True,
              a=1, b=0, loss_type='xie', beta=1, gamma=1):

    n, c, h, w = inputs.size()

    t = np.zeros((1, c, h, w))
    for i in range(n):
        t += targets[i, :, :, :].cpu().data.numpy()

    pos = (t == 1).sum()
    neg = (t == 0).sum()

    alpha = (neg + 1) / (pos + 1)
    # alpha = neg / pos
    #################  alpha 是一个 batch 中负样本和正样本的比例.   lk #################
    if loss_type =='ce':
        pos_weight = 1
    if loss_type == 'xie':
        pos_weight = alpha

    pos_weight = torch.Tensor((torch.ones(1) * pos_weight))         ##### lk 01.07

    if cuda:
        pos_weight = pos_weight.cuda()
        pos_weight = pos_weight.detach()

    # loss = a * nn.BCEWithLogitsLoss(weight=None, size_average=False,
    #                                 pos_weight=pos_weight)(inputs, targets) \
    #        - b * jacc_coef(F.sigmoid(inputs), targets)

    loss = a * nn.BCEWithLogitsLoss(weight=None, reduction='mean',
                                    pos_weight=pos_weight)(inputs, targets) \
           - b * jacc_coef(F.sigmoid(inputs), targets)

    return loss

#########################################################################################
def parse_args():
    parser = argparse.ArgumentParser(description='Train CarNet for different args')
    parser.add_argument('-d', '--dataset', type=str, choices=cfg.config.keys(),
        default='Sun520_aug', help='The dataset to train')
    parser.add_argument('--param-dir', type=str,
        default='Sun520_aug_CarNet',
        help='the directory to store the params')
    parser.add_argument('--lr', dest='base_lr', type=float, default=3e-4,
        help='the base learning rate of model')
    # parser.add_argument('-m', '--momentum', type=float, default=0.9,
    #     help='the momentum')
    parser.add_argument('-c', '--cuda', action='store_true',
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('--weight-decay', type=float, default=0,
        help='the weight_decay of net, default is 0.0002')
    parser.add_argument('-p', '--pretrain', type=str, default=None,
        help='init net from pretrained model default is vgg16.pth')
    parser.add_argument('--max-iter', type=int, default=1000*15,
        help='max iters to train network, '
            'default is 1000*15 for Sun520, 500*30 for BJN260, 750*20 for Rain365, 650*30 for Crack360')
    parser.add_argument('--iter-size', type=int, default=1,
        help='iter size equal to the batch size, default 10')
    parser.add_argument('--average-loss', type=int, default=50,
        help='smoothed loss, default is 50')
    parser.add_argument('-s', '--snapshots', type=int, default=1000*5,
        help='how many iters to store the params, '
             'default is 1000*5 for Sun520, 500*5 for BJN260, 750*5 for Rain365, 650*5 for Crack360')
    parser.add_argument('--step-size', type=int, default=650*30,
        help='the number of iters to decrease the learning rate, '
             'default is 1000*10 for Sun520, 500*25 for BJN260, 750*15 for Rain365, 650*30 for Crack360')
    parser.add_argument('--display', type=int, default=100,
        help='how many iters display one time, default is 20; 1000')
    parser.add_argument('-b', '--balance', type=float, default=1,
        help='the parameter to balance the neg and pos, default is 1.1')
    parser.add_argument('-l', '--log', type=str, default='log.txt',
        help='the file to store log, default is log.txt')
    parser.add_argument('--batch-size', type=int, default=2,
        help='batch size of one iteration, default 1')
    parser.add_argument('--crop-size', type=int, default=None,
        help='the size of image to crop, default not crop, but crop 512 for Crack 360')
    parser.add_argument('--complete-pretrain', type=str, default=None,
        help='finetune on the complete_pretrain, default None')
    parser.add_argument('--gamma', type=float, default=0.1,
        help='the decay of learning rate, default 0.1')

    parser.add_argument('--a', type=float, default=1,
        help='the coefficient of wce, default 1')
    parser.add_argument('--b', type=float, default=0,
        help='the coefficient of jaccard, default 0')
    parser.add_argument('--type', type=str, default='ce',
        help='the type of loss function, default xie')
    parser.add_argument('--beta', type=float, default=1,
        help='Fine-tune the proportion of positive and negative samples, default 1')
    parser.add_argument('--lgamma', type=float, default=1,
        help='adjust the proportion of positive and negative samples, default 1')

    return parser.parse_args()

def train(model, args):
    data_root = cfg.config[args.dataset]['data_root']
    data_lst = cfg.config[args.dataset]['data_lst']

    train_img = CrackData(data_root, data_lst, crop_size=args.crop_size)
    trainloader = torch.utils.data.DataLoader(train_img,
        batch_size=args.batch_size, shuffle=True, num_workers=8)

    params_dict = dict(model.named_parameters())
    base_lr = args.base_lr
    weight_decay = args.weight_decay
    logger = args.logger
    params = []

    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    start_step = 1
    mean_loss = []
    cur = 0
    pos = 0
    data_iter = iter(trainloader)

    iter_per_epoch = len(trainloader)
    logger.info('*'*40)
    logger.info('train images in all are %d ' % (iter_per_epoch * args.batch_size))
    logger.info('the batch size is %d ' % (args.iter_size * args.batch_size))
    logger.info('every epoch needs to iterate  %d ' % iter_per_epoch)
    logger.info('*'*40)

    start_time = time.time()
    if args.cuda:
        model.cuda()

    model.train()
    batch_size = args.iter_size * args.batch_size
    for step in range(start_step, args.max_iter + 1):
        optimizer.zero_grad()
        batch_loss = 0
        for i in range(args.iter_size):
            if cur == iter_per_epoch:
                cur = 0
                data_iter = iter(trainloader)
            images, labels = next(data_iter)
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)
            out = model(images)
            loss = 0

            ##########  single output #########################
            # loss += cross_entropy_loss2d(out, labels, args.cuda, args.balance) / batch_size

            loss += wcet_jacc(inputs=out, targets=labels, cuda=True,
                              a=args.a, b=args.b, loss_type=args.type, beta=args.beta, gamma=args.lgamma) / batch_size

            loss.backward()
            # batch_loss += loss.data[0]      #####lk
            batch_loss += loss.item()   #####lk
            cur += 1
        # update parameter
        optimizer.step()
        if len(mean_loss) < args.average_loss:
            mean_loss.append(batch_loss)
        else:
            mean_loss[pos] = batch_loss
            pos = (pos + 1) % args.average_loss
        if step % args.step_size == 0:
            adjust_learning_rate(optimizer, step, args.step_size, args.gamma)
        if step % args.snapshots == 0:
            torch.save(model.state_dict(), '%s/CarNet_%d.pth' % (args.param_dir, step))
            # state = {'step': step+1,'param':model.state_dict(),'solver':optimizer.state_dict()}
            # torch.save(state, '%s/CarNet_%d.pth.tar' % (args.param_dir, step))
        if step % args.display == 0:
            tm = time.time() - start_time
            # logger.info('iter: %d, lr: %e, loss: %f, time using: %f(%fs/iter)' % (step,
            #     optimizer.param_groups[0]['lr'], np.mean(mean_loss), tm, tm/args.display))

            logger.info('iter: %d, lr: %e, loss: %f, time using: %f(%fs/iter)' % (step,
                optimizer.param_groups[0]['lr'], np.mean(mean_loss), tm, tm/args.display))

            start_time = time.time()

def main():
    args = parse_args()

    args.cuda = True    ######### lk
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger = log.get_logger(args.log)
    args.logger = logger
    logger.info('*'*80)
    logger.info('the args are the below')
    logger.info('*'*80)
    for x in args.__dict__:
        logger.info(x+','+str(args.__dict__[x]))
    logger.info(cfg.config[args.dataset])
    logger.info('*'*80)

    if not os.path.exists(args.param_dir):
        os.mkdir(args.param_dir)
    # torch.manual_seed(int(time.time()))
    torch.manual_seed(seed=7)  #### lk
    torch.cuda.manual_seed(seed=7)
    np.random.seed(seed=7)
    torch.backends.cudnn.deteministic=True  #### lk

    model = CarNet34(
        Encoder=Encoder_v0_762,
        dp=DownsamplerBlock,
        block=BasicBlock_encoder,
        channels=[3, 16, 64, 128, 256],
        decoder_block=non_bottleneck_1d_2,
        num_classes=1
    )

    if args.complete_pretrain:
        model.load_state_dict(torch.load(args.complete_pretrain))

    logger.info(model)

    train(model, args)

if __name__ == '__main__':
    main()


