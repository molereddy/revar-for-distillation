import sys, time, torch, random, argparse, json, os
import itertools
from collections import namedtuple
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import torch.utils.data as data
from typing import Type, Any, Callable, Union, List, Optional
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt

from models import get_instant_weight_model
from model_dict import get_model_from_name
from utils import get_model_infos
from log_utils import AverageMeter, time_string, convert_secs2time
from starts import prepare_logger, prepare_seed
from get_dataset_with_transform import get_datasets
from DiSK import *

def main(args):

    args.save_dir = args.save_dir
    print(args)

    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    torch.set_num_threads(args.workers)

    criterion = nn.CrossEntropyLoss()

    train_data, test_data, xshape, class_num = get_datasets(
        args.dataset, args.data_path, args.cutout_length
    )
    train_data, valid_data = data.random_split(train_data, [len(train_data)-len(train_data)//10, 
                                                            len(train_data)//10])
    shuffle = args.shuffle == 1
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    tmp_train_loader = None

    args.class_num = class_num
    logger = prepare_logger(args)
    prepare_seed(args.rand_seed)
    
    logger.log(args.__str__())
    logger.log("Train:{}\t, Valid:{}\t, Test:{}\n".format(len(train_data),
                                                          len(valid_data),
                                                          len(test_data)))

    Arguments = namedtuple("Configure", ('class_num','dataset')  )
    md_dict = { 'class_num' : class_num, 'dataset' : args.dataset }
    model_config = Arguments(**md_dict)


    # STUDENT
    student_model = get_model_from_name( model_config, args.model_name )
    model_name = args.model_name

    logger.log(("Student: {}".format(model_name)))
    # ce_ptrained_path = "/home/anmolreddy/pretrained/disk-CE-cifar100-ResNet10_s-model_best.pth.tar"
    ce_ptrained_path = "/home/anmolreddy/ce_results/CE_with_seed-{}_cycles-{}_{}-{}"\
                        "model_best.pth.tar".format(args.rand_seed,
                                                    args.sched_cycles,
                                                    args.dataset,
                                                    args.model_name)
    assert Path().exists(), "Cannot find the initialization file : {:}".format(ce_ptrained_path)
    logger.log("using pretrained student model from {}".format(ce_ptrained_path))
    base_checkpoint = torch.load(ce_ptrained_path)
    student_model.load_state_dict(base_checkpoint["base_state_dict"])
    
    student_model = student_model.cuda()
    network = student_model


    # TEACHER
    md_dict = { 'class_num' : class_num, 'dataset' : args.dataset }
    global_model_config = model_config = Arguments(**md_dict)
    teacher_model = get_model_from_name(global_model_config, args.global_model_name)
    global_model_name = args.global_model_name
    teach_PATH = "/home/anmolreddy/pretrained/teacher_seed-{}_{}-{}"\
                    "model_best.pth.tar".format(args.rand_seed,
                                                args.dataset,
                                                args.global_model_name)
    teach_checkpoint = torch.load(teach_PATH)
    teacher_model.load_state_dict(teach_checkpoint['base_state_dict'])
    teacher_model = teacher_model.cuda()
    teacher = teacher_model

    s_idx, t_idx = 1, 1
    #if len(student_model.conv_channels)>1: s_idx=2
    #if len(teacher_model.conv_channels)>1: t_idx=2
    routingNet = get_instant_weight_model(
            num_s_ft = student_model.xchannels[-1],
            num_t_ft = teacher_model.xchannels[-1],
            num_s_conv_ft = student_model.conv_channels[-s_idx],
            num_t_conv_ft = teacher_model.conv_channels[-t_idx],
            routing_name=args.routing_name, 
            n_labels=class_num, )


    if args._ckpt != "" and len(args._ckpt)>3:
        state = torch.load(args._ckpt)
        #teacher.load_state_dict( state['global_state_dict'] )
        network.load_state_dict( state['base_state_dict'] )
        #routingNet.load_state_dict( state['routing_state_dict'] )

    flop, param = get_model_infos(teacher_model, xshape)
    args.global_flops = flop 
    logger.log("model information : {:}".format(teacher_model.get_message()))
    logger.log("-" * 50)
    logger.log(
        "[Teacher]Params={:.2f} MB, FLOPs={:.2f} M ... = {:.2f} G".format(
            param, flop, flop / 1e3
        )
    )

    flop, param = get_model_infos(student_model, xshape)
    args.base_flops = flop 
    logger.log("model information : {:}".format(student_model.get_message()))
    logger.log("-" * 50)
    logger.log(
        "[Student]Params={:.2f} MB, FLOPs={:.2f} M ... = {:.2f} G".format(
            param, flop, flop / 1e3
        )
    )

    logger.log("-" * 50)
    logger.log("train_data : {:}".format(train_data))
    logger.log("valid_data : {:}".format(valid_data))

    test_loss, test_acc1, test_acc5 = evaluate_model( network, test_loader, criterion, args.eval_batch_size )
    logger.log(
        "***{:s}*** before training [Student(CE)] Test loss = {:.6f}, accuracy@1 = {:.2f}, accuracy@5 = {:.2f}, error@1 = {:.2f}, error@5 = {:.2f}".format(
            time_string(),
            test_loss,
            test_acc1,
            test_acc5,
            100 - test_acc1,
            100 - test_acc5,
        )
    )

    test_loss, test_acc1, test_acc5 = evaluate_model( teacher, test_loader, criterion, args.eval_batch_size )
    logger.log(
        "***{:s}*** [Teacher] Test loss = {:.6f}, accuracy@1 = {:.2f}, accuracy@5 = {:.2f}, error@1 = {:.2f}, error@5 = {:.2f}".format(
            time_string(),
            test_loss,
            test_acc1,
            test_acc5,
            100 - test_acc1,
            100 - test_acc5,
        )
    )

    epochs = args.epochs
    resume_checkpoint = args.resume_ckpt

    main_train_eval_loop( logger, args, network, teacher, routingNet, 
          train_loader, valid_loader, 
          model_name, global_model_name, resume_checkpoint=resume_checkpoint, 
          start_epoch=0, epochs=epochs, steps = 10, )

    test_loss, test_acc1, test_acc5 = evaluate_model( network, test_loader, criterion, args.eval_batch_size )
    logger.log(
        "***{:s}*** [Post-train] [Student] Test loss = {:.6f}, accuracy@1 = {:.2f}, accuracy@5 = {:.2f}, error@1 = {:.2f}, error@5 = {:.2f}".format(
            time_string(),
            test_loss,
            test_acc1,
            test_acc5,
            100 - test_acc1,
            100 - test_acc5,
        )
    )



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train a classification model on typical image classification datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--negative', action='store_true')

    parser.add_argument("--KD_temperature", type=float, default=4, help="The temperature parameter in knowledge distillation.")

    parser.add_argument('--max_ce', type=float, default=5., help='number of epochs per alternates')
    parser.add_argument("--KD_temperature_s", type=float, default=3, help="The temperature parameter in knowledge distillation.")


    parser.add_argument("--_ckpt", type=str, default='', help="The path to the model checkpoint")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="The path to the model checkpoint")
    parser.add_argument("--optim_config", type=str, default='./configs/opts/CIFAR-E300-W5-L1-COS.config', help="The path to the optimizer configuration")
    parser.add_argument("--model_ckpt", type=str, 
        #default='./output/search-shape/TAS-INFER-cifar10-C010-ResNet32TwoPFiveM-NAS-KDTT/checkpoint/seed-2006-basic.pth', 
        default='./ckpt/C010-ResNet32TwoPFiveM-NAS-KDT-seed-2006-basic.pth', help="The path to the model checkpoint")
    parser.add_argument("--global_model_ckpt", type=str, 
        #default='./output/search-shape/TAS-INFER-cifar10-C010-ResNet32-KDT/checkpoint/seed-2010-best.pth ', 
        default='./ckpt/C010-ResNet32-KDT-seed-2010-best.pth ', help="The path to the global model checkpoint")
    parser.add_argument("--global_model_name", type=str, default='ResNet32', help="The path to the global model configuration")
    parser.add_argument("--model_name", type=str, default='ResNet32TwoPFiveM-NAS', help="The path to the model configuration")
    parser.add_argument("--procedure", type=str, default="opt-sc-oracle-routing-joint-scheme", help="The procedure basic prefix.")
    parser.add_argument("--model_source", type=str, default="normal", help="The source of model defination.")
    parser.add_argument("--extra_model_path", type=str, default=None, help="The extra model ckp file (help to indicate the searched architecture).")

    # Data Generation
    parser.add_argument("--dataset", type=str, default='cifar10', help="The dataset name.")
    parser.add_argument("--data_path", type=str, default='./data/', help="The dataset name.")
    parser.add_argument("--cutout_length", type=int, default=16, help="The cutout length, negative means not use.")

    # 1 --> training gating
    # 2 --> training jointly all components
    parser.add_argument('--train_scheme', type=int, default=2, help="Joint training or only training the gating")

    # 1 --> Use logits etc. features from base model
    # 2 --> Use Resnet18 model 
    parser.add_argument('--gate_arch', type=int, default=1, help="Gate Architecture")
    parser.add_argument('--routing_name', type=str, default='default', help="Gate Architecture")
    parser.add_argument("--gate_label", type=int, default=1, help="Gate labelling strategy")
    parser.add_argument("--gate_conf", type=int, default=1, help="Gate labelling confidence strategy")

    # Printing
    parser.add_argument("--print_freq", type=int, default=100, help="print frequency (default: 200)")
    parser.add_argument("--print_freq_eval", type=int, default=100, help="print frequency (default: 200)")
    # Checkpoints
    parser.add_argument("--eval_frequency", type=int, default=1,help="evaluation frequency (default: 200)")
    parser.add_argument("--save_dir", type=str, help="Folder to save checkpoints and log.", default='./trained/')
    # Acceleration
    parser.add_argument("--workers", type=int, default=8, help="number of data loading workers (default: 8)")
    # Random Seed
    parser.add_argument("--rand_seed", type=int, help="base model seed")
    parser.add_argument("--global_rand_seed", type=int, default=-1, help="global model seed")
    #add_shared_args(parser)

    # Optimization options
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=200, help="Batch size for training.")

    parser.add_argument('--log-dir', default='./log', help='tensorboard log directory')
    parser.add_argument('--checkpoint-dir', default='./checkpoint', help='checkpoint file format')
    parser.add_argument('-p', '--print-freq', default=50, type=int, metavar='N', help='print frequency (default: 10)')
    # architecture setting
    parser.add_argument('--cov', type=float, default=0.38, help='target 1-coverage term in the formulation.')
    #parser.add_argument('--g_denom', type=float, default=1., help='denominator in the balanced routing loss formulation.')
    parser.add_argument('--g_denom', type=float, default=0.2, help='denominator in the balanced routing loss formulation.')
    #parser.add_argument('--strategy', type=int, default=1, help='number of epochs per alternates')
    parser.add_argument('--strategy', type=int, default=2, help='number of epochs per alternates')
    # weights for only hybrid, hybrid_kd
    parser.add_argument('--shuffle', type=int, default=1, help='number of epochs per alternates')
    parser.add_argument('--base_strategy', type=int, default=1, help='number of epochs per alternates')
    parser.add_argument('--n_base_strategy', type=int, default=1, help='number of epochs per alternates')
    parser.add_argument('--penalty', type=int, default=1, help='number of epochs per alternates')
    parser.add_argument('--use_kl', type=int, default=0, help='number of epochs per alternates')
    parser.add_argument('--use_oracle', type=int, default=0, help='number of epochs per alternates')
    parser.add_argument('--use_prob', type=int, default=0, help='number of epochs per alternates')
    parser.add_argument('--use_g_clf', type=int, default=0, help='number of epochs per alternates')
    parser.add_argument('--use_g_y_routing', type=int, default=0, help='number of epochs per alternates')
    parser.add_argument('--use_prob_wts', type=int, default=0, help='number of epochs per alternates')
    parser.add_argument('--use_auxiliary', type=int, default=1, help='number of epochs per alternates')
    parser.add_argument('--use_only_bn', type=int, default=0, help='number of epochs per alternates')
    parser.add_argument('--use_alt_min', type=int, default=0, help='number of epochs per alternates')
    parser.add_argument('--restart_s_learn', type=int, default=0, help='number of epochs per alternates')
    parser.add_argument('--restart_g_learn', type=int, default=0, help='number of epochs per alternates')
    parser.add_argument('--alt_min_every_n_epochs', type=int, default=1, help='number of epochs per alternates')
    parser.add_argument('--tau', type=float, default=0.8, help='number of epochs per alternates')
    parser.add_argument('--tau2', type=float, default=0.8, help='number of epochs per alternates')
    parser.add_argument('--l_sg', type=float, default=0.6, help='number of epochs per alternates')
    parser.add_argument('--l_aux', type=float, default=1., help='number of epochs per alternates')
    parser.add_argument('--l_args_wts', type=float, default=1., help='number of epochs per alternates')
    parser.add_argument('--l_kl', type=float, default=1., help='number of epochs per alternates')
    parser.add_argument('--l_nll', type=float, default=1., help='number of epochs per alternates')
    parser.add_argument('--l_oracle', type=float, default=1., help='number of epochs per alternates')
    parser.add_argument('--l_g_clf', type=float, default=0.5, help='number of epochs per alternates')

    parser.add_argument('--b_add_sparsity_alt_min', type=int, default=0, help='learning rate for a single GPU')

    parser.add_argument('--primal_budget_update', type=int, default=0, help='learning rate for a single GPU')
    parser.add_argument('--budget_g', type=float, default=0.0, help='learning rate for a single GPU')
    parser.add_argument('--budget_Ti', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--budget_g_min', type=float, default=0.0, help='learning rate for a single GPU')
    parser.add_argument('--budget_g_max', type=float, default=0.0, help='learning rate for a single GPU')
    parser.add_argument('--budget_g_gamma', type=float, default=0.9, help='learning rate for a single GPU')

    parser.add_argument('--rho', type=float, default=0.1, help='learning rate for a single GPU')
    parser.add_argument('--lmbda_dual', type=float, default=0.4, help='learning rate for a single GPU')
    parser.add_argument('--lmbda', type=float, default=0.4, help='learning rate for a single GPU')
    #parser.add_argument('--lmbda_dual', type=float, default=0.4, help='learning rate for a single GPU')
    parser.add_argument('--lmbda_min', type=float, default=0.01, help='learning rate for a single GPU')
    parser.add_argument('--lmbda_adaptive', type=int, default=0, help='learning rate for a single GPU')
    parser.add_argument('--Ti', type=int, default=100, help='number of epochs to train')


    parser.add_argument('--percentile_t_prob_th', type=float, default=0.6, help='number of epochs per alternates')

    # ce, kd, hybrid, hybrid_kd
    parser.add_argument('--base_method', type=str, default='hybrid', help='number of epochs per alternates')
    parser.add_argument('--topK', type=int, default=3, help='number of epochs per alternates')
    parser.add_argument('--kt', type=int, default=3, help='number of epochs per alternates')
    parser.add_argument('--kg', type=int, default=3, help='number of epochs per alternates')
    parser.add_argument('--temp', type=int, default=4, help='number of epochs per alternates')
    parser.add_argument('--base_opt_type', default='adam', type=str)
    parser.add_argument('--s_lr', type=float, default=0.001, help='learning rate for base')
    parser.add_argument('--s_iters', type=int, default=5005, help='batches for base optimization')

    parser.add_argument('--global_opt_type', default='adam', type=str)
    parser.add_argument('--t_lr', type=float, default=0.0001, help='learning rate for global')
    parser.add_argument('--t_iters', type=int, default=5005, help='batches for global optimization')

    parser.add_argument('--routing_opt_type', default='adam', type=str)
    parser.add_argument('--g_lr', type=float, default=0.01, help='learning rate for gate')
    parser.add_argument('--g_lr_init', type=float, default=0.01, help='learning rate for gate')
    parser.add_argument('--g_iters', type=int, default=505, help='batches for gate optimization')

    parser.add_argument('--n_parts', type=int, default=1,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--init_epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0125,
                        help='learning rate for a single GPU')
    parser.add_argument('--warmup-epochs', type=float, default=5,
                        help='number of warmup epochs')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--wd', type=float, default=0.00001,  help='weight decay')
    parser.add_argument('--eps', type=float, default=0.05,  help='weight decay')
    parser.add_argument('--r_wd', type=float, default=0.00001,  help='weight decay')
    parser.add_argument('--lr_type', type=str, default='cosine')

    parser.add_argument('--sched_cycles', type=int, default=4,  help='How many times cosine cycles for scheduler')
    parser.add_argument('--file_name', type=str, default="",  help='file_name')
    args = parser.parse_args()


    assert( args.base_method in [ 'ce', 'kd', 'hybrid', 'hybrid_kd', 'hybrid_kd_inst' ] )

    assert( args.train_scheme in [1,2] )
    assert( args.gate_arch in [1,2] )
    assert( args.gate_conf in [1,2,3,4] )
    assert( args.gate_label in [1,2,3,4,5,6,7,8,9] )
    print('train_scheme =', args.train_scheme, ' --gate_arch=', args.gate_arch, ' -- gate_label', args.gate_label, '  -- gate_conf', args.gate_conf)
    #assert(1==2)
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 10)
    if (args.file_name is None or args.file_name == ""):
        args.file_name = "DiSK_seed-{}_cycles-{}_T-{}_{}".format(
                                                            args.rand_seed, 
                                                            args.sched_cycles,
                                                            int(args.KD_temperature_s),
                                                            args.epochs)
    assert args.save_dir is not None, "save-path argument can not be None"
    #return args

    main(args)





