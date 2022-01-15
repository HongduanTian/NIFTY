import argparse

def argument_parser():
    '''
    Get arguments for training and evaluation.
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=4, help='seed for random')
    parser.add_argument('--data_path', type=str, default='./german', help='data path')
    parser.add_argument('--model', type=str, default='niftygcn', help='model options: gcn or niftygcn')
    parser.add_argument('--num_hidden', type=int, default=16, help='num of units of hidden layer')
    parser.add_argument('--num_class', type=int, default=2, help='num of classes')
    parser.add_argument('--dropout', type=float, default=0.2, help='drop rate of Dropout layer')
    parser.add_argument('--data_perturb', type=float, default=0.1, help='prob of data perturbation')
    parser.add_argument('--struc_perturb', type=float, default=1e-3, help='drop prob of structure perturbation')
    parser.add_argument('--num_epoch', type=int, default=1000, help='num of epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate of optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--coef', type=float, default=0.3, help='coefficient of loss in nifty')
    parser.add_argument('--lipschitz', help='whether use lipschitz norm', action='store_true')
    parser.add_argument('--sim', help='whether use similarity loss', action='store_true')
    #parser.add_argument('--lr_scale', type=float, default=1.0, help='scale for learning rate')

    return parser