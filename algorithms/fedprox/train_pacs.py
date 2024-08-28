import os
import argparse
from utils.log_utils import *
from network.get_network import GetNetwork
from torch.utils.tensorboard.writer import SummaryWriter
from data.pacs_dataset import PACS_FedDG
from data.officehome_dataset import OfficeHome_FedDG
from data.vlcs_dataset import VLCS_FedDG
from utils.classification_metric import Classification 
import torch
from utils.fed_merge import Cal_Weight_Dict, FedAvg, FedUpdate
from utils.trainval_func import site_train, site_evaluation, SaveCheckPoint
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
from network.FedOptimizer.FedProx import FedProx
from tqdm import tqdm
import sys
from configs.default import *

from absl import app
from absl import flags
from utils import flags_cc
from pytorch_lightning.loggers import WandbLogger

FLAGS = flags.FLAGS

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='vlcs', choices=['pacs','officehome', 'vlcs'], help='Name of dataset')
    parser.add_argument("--model", type=str, default='resnet18',
                        choices=['resnet18', 'resnet18_rsc', 'agg', 'mobilenetv2', 
                                 'mobilenetv1', 'ccnet', 'vittiny'], help='model name')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--no-pretrain', dest='pretrain', action='store_false')
    parser.add_argument('--agg_ckpt', dest='agg_ckpt', type=str, default=None)
    parser.add_argument("--test_domain", type=str, default='v',
                        choices=['p', 'a', 'c', 's', 'r', 'v','l','c','s'], help='the domain name for testing')
    parser.add_argument('--num_classes', help='number of classes default 7', type=int, default=7)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=256)
    parser.add_argument('--local_epochs', help='epochs number', type=int, default=5)
    parser.add_argument('--comm', help='epochs number', type=int, default=10)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.0003)
    parser.add_argument('--mu', help='mu for FedProx', type=float, default=0.1)
    parser.add_argument("--lr_policy", type=str, default='one_cycle', choices=['step'],
                        help="learning rate scheduler policy")
    parser.add_argument('--note', help='note of experimental settings', type=str, default='fedavg')
    parser.add_argument('--display', help='display in controller', action='store_true') # 默认false 即不展示
    parser.add_argument('--flagfile', help='flagfile', type=str, default=None) 
    parser.add_argument('--comment', help='comment', type=str, default='no_comment')
  
    return parser.parse_args()


def GetFedModel(args, num_classes, is_train=True):
    global_model, feature_level = GetNetwork(args, args.num_classes, True)
    global_model = global_model.cuda()
    model_dict = {}
    optimizer_dict = {}
    scheduler_dict = {}

    if args.dataset == 'pacs':
        domain_list = pacs_domain_list
    elif args.dataset == 'officehome':
        domain_list = officehome_domain_list
    elif args.dataset == 'domainNet':
        domain_list = domainNet_domain_list
    elif args.dataset == 'terrainc':
        domain_list = terra_incognita_list
    elif args.dataset == 'vlcs':
        domain_list = vlcs_domain_list

    for domain_name in domain_list:
        model_dict[domain_name], _ = GetNetwork(args, num_classes, is_train)
        model_dict[domain_name] = model_dict[domain_name].cuda()
        optimizer_dict[domain_name] = FedProx(model_dict[domain_name].parameters(), lr=args.lr, momentum=0.9,
                                                      weight_decay=5e-4, mu=args.mu)
        optimizer_dict[domain_name].update_old_init(global_model.parameters()) # 保存初始化的wt
        total_epochs = args.local_epochs * args.comm
        if args.lr_policy == 'step':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.StepLR(optimizer_dict[domain_name], step_size=args.local_epochs * args.comm, gamma=0.1)
        elif args.lr_policy == 'one_cycle':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.OneCycleLR(
                optimizer_dict[domain_name],
                max_lr=args.lr,
                epochs=total_epochs,
                steps_per_epoch=45000//args.batch_size)

    return global_model, model_dict, optimizer_dict, scheduler_dict


def main():
    '''log part'''
    file_name = 'fedprox_'+os.path.split(__file__)[1].replace('.py', '')
    args = get_argparse()
    args.FLAGS = FLAGS
    wandb_logger = WandbLogger(project="FEDPARTWHOLE", name=FLAGS.exp_name)
    wandb_logger.experiment.config.update(FLAGS)
    log_dir, tensorboard_dir = Gen_Log_Dir(args, file_name=file_name)
    log_ten = SummaryWriter(log_dir=tensorboard_dir)
    log_file = Get_Logger(file_name=log_dir + 'train.log', display=args.display)
    log_file.info(f'comment: {args.comment}')
    Save_Hyperparameter(log_dir, args)
    
    '''dataset and dataloader'''
    if args.dataset == 'officehome':
        dataobj = OfficeHome_FedDG(test_domain=args.test_domain, batch_size=args.batch_size)
    elif args.dataset == 'pacs':
        dataobj = PACS_FedDG(test_domain=args.test_domain, batch_size=args.batch_size)
    elif args.dataset == 'vlcs':
        dataobj = VLCS_FedDG(test_domain=args.test_domain, batch_size=args.batch_size)
    dataloader_dict, dataset_dict = dataobj.GetData()
    
    '''model'''
    metric = Classification()
    global_model, model_dict, optimizer_dict, scheduler_dict = GetFedModel(args, args.num_classes)
    weight_dict = Cal_Weight_Dict(dataset_dict, site_list=dataobj.train_domain_list)
    FedUpdate(model_dict, global_model)
    best_val = 0.
    for i in range(args.comm+1):
        FedUpdate(model_dict, global_model)
        for domain_name in dataobj.train_domain_list:
            site_train(i, domain_name, args, model_dict[domain_name], optimizer_dict[domain_name], 
                       scheduler_dict[domain_name],dataloader_dict[domain_name]['train'], log_ten, metric)
            
            site_evaluation(i, domain_name, args, model_dict[domain_name], dataloader_dict[domain_name]['val'], log_file, log_ten, metric, note='before_fed')
        FedAvg(model_dict, weight_dict, global_model)
        
        fed_val = 0.
        for domain_name in dataobj.train_domain_list:
            results_dict = site_evaluation(i, domain_name, args, global_model, dataloader_dict[domain_name]['val'], log_file, log_ten, metric)
            fed_val+= results_dict['acc']*weight_dict[domain_name]
        # val 结果
        if fed_val >= best_val:
            best_val = fed_val
            SaveCheckPoint(args, global_model, args.comm, os.path.join(log_dir, 'checkpoints'), note='best_val_model')
            for domain_name in dataobj.train_domain_list:
                SaveCheckPoint(args, model_dict[domain_name], args.comm, os.path.join(log_dir, 'checkpoints'), note=f'best_val_{domain_name}_model')
                
            log_file.info(f'Model saved! Best Val Acc: {best_val*100:.2f}%')
        site_evaluation(i, args.test_domain, args, global_model, dataloader_dict[args.test_domain]['test'], log_file, log_ten, metric, note='test_domain')
        
    SaveCheckPoint(args, global_model, args.comm, os.path.join(log_dir, 'checkpoints'), note='last_model')
    for domain_name in dataobj.train_domain_list: 
        SaveCheckPoint(args, model_dict[domain_name], args.comm, os.path.join(log_dir, 'checkpoints'), note=f'last_{domain_name}_model')
    
if __name__ == '__main__':
    #app.run(main)    
    FLAGS(sys.argv, known_only=True)
    main()