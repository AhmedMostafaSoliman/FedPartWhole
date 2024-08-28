import os
import sys
import argparse
from utils.log_utils import *
from network.get_network import GetNetwork
from torch.utils.tensorboard.writer import SummaryWriter
from data.pacs_dataset import PACS_FedDG
from data.officehome_dataset import OfficeHome_FedDG
from data.vlcs_dataset import VLCS_FedDG
from data.officehome_dataset import OfficeHome_FedDG
from utils.classification_metric import Classification
import torch
from utils.fed_merge import Cal_Weight_Dict, FedAvg, FedUpdate
from network.FedOptimizer.Scaffold import *
import torch.nn.functional as F
from tqdm import tqdm
from utils.trainval_func import site_evaluation, SaveCheckPoint
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
    parser.add_argument('--lr', help='learning rate', type=float, default=0.0004)
    parser.add_argument("--lr_policy", type=str, default='step', choices=['step'],
                        help="learning rate scheduler policy")
    parser.add_argument('--note', help='note of experimental settings', type=str, default='fedavg')
    parser.add_argument('--display', help='display in controller', action='store_true')
    parser.add_argument('--flagfile', help='flagfile', type=str, default=None) 
    parser.add_argument('--comment', help='comment', type=str, default='no_comment')
    return parser.parse_args()

def epoch_site_train(epochs, site_name, model, optimzier, scheduler, c_ci, dataloader, log_ten, metric):
    model.train()
    for i, data_list in enumerate(dataloader):
        imgs, labels, domain_labels, embeds, indices = data_list
        imgs = imgs.cuda()
        labels = labels.cuda()
        embeds = embeds.cuda()
        domain_labels = domain_labels.cuda()
        optimzier.zero_grad()
        if model.__class__.__name__ == 'ccnet':
            output, _ = model(imgs, levels=embeds) #glom
        elif model.__class__.__name__ == 'Agglomerator':
            output, _ = model(imgs)
        else:
            output = model(imgs)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        optimzier.step(c_ci) 
        log_ten.add_scalar(f'{site_name}_train_loss', loss.item(), epochs*len(dataloader)+i)
        metric.update(output, labels)
    
    log_ten.add_scalar(f'{site_name}_train_acc', metric.results()['acc'], epochs)
    scheduler.step()
    
def site_train(comm_rounds, site_name, args, model, optimizer, scheduler, c_ci, dataloader, log_ten, metric):
    tbar = tqdm(range(args.local_epochs))
    for local_epoch in tbar:
        tbar.set_description(f'{site_name}_train')
        epoch_site_train(comm_rounds*args.local_epochs + local_epoch, site_name, model, optimizer, scheduler, c_ci,  dataloader, log_ten, metric)


def GetFedModel(args, num_classes, is_train=True):
    global_model, feature_level = GetNetwork(args, args.num_classes, True)
    global_model = global_model.cuda()
    model_dict = {}
    optimizer_dict = {}
    scheduler_dict = {}
    ci_dict = {}
    c = GenZeroParamList(global_model)

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
        
        ci_dict[domain_name] = GenZeroParamList(model_dict[domain_name])
        
        optimizer_dict[domain_name] = Scaffold(model_dict[domain_name].parameters(), lr=args.lr, momentum=0.9,
                                                      weight_decay=5e-4)
        total_epochs = args.local_epochs * args.comm
        if args.lr_policy == 'step':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.StepLR(optimizer_dict[domain_name], step_size=args.local_epochs * args.comm, gamma=0.1)
        elif args.lr_policy == 'one_cycle':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.OneCycleLR(
                optimizer_dict[domain_name],
                max_lr=args.lr,
                epochs=total_epochs,
                steps_per_epoch=45000//args.batch_size)
    return global_model, model_dict, optimizer_dict, scheduler_dict, ci_dict, c

    
def main():
    file_name = 'scaffold_'+os.path.split(__file__)[1].replace('.py', '')
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
    global_model, model_dict, optimizer_dict, scheduler_dict, ci_dict, c = GetFedModel(args, args.num_classes)
    weight_dict = Cal_Weight_Dict(dataset_dict, site_list=dataobj.train_domain_list)
    
    best_val = 0.
    for i in range(args.comm+1):
        FedUpdate(model_dict, global_model)
        for domain_name in dataobj.train_domain_list:
            c_ci = ListMinus(c, ci_dict[domain_name])
            K = len(dataloader_dict[domain_name]['train']) * args.local_epochs
            site_train(i, domain_name, args, model_dict[domain_name], optimizer_dict[domain_name], 
                       scheduler_dict[domain_name], c_ci, dataloader_dict[domain_name]['train'], log_ten, metric)
            
            site_evaluation(i, domain_name, args, model_dict[domain_name], dataloader_dict[domain_name]['val'], log_file, log_ten, metric, note='before_fed')
        
            ci_dict[domain_name] = UpdateLocalControl(c, ci_dict[domain_name], global_model,  model_dict[domain_name], K)

        c = UpdateServerControl(c,ci_dict, weight_dict)
        
        FedAvg(model_dict, weight_dict, global_model)  
        
        fed_val = 0.
        for domain_name in dataobj.train_domain_list:
            results_dict = site_evaluation(i, domain_name, args, global_model, dataloader_dict[domain_name]['val'], log_file, log_ten, metric)
            fed_val+= results_dict['acc']*weight_dict[domain_name]
        if fed_val >= best_val:
            best_val = fed_val
            SaveCheckPoint(args, global_model, args.comm, os.path.join(log_dir, 'checkpoints'), note='best_val_model')
            for domain_name in dataobj.train_domain_list: # local model
                SaveCheckPoint(args, model_dict[domain_name], args.comm, os.path.join(log_dir, 'checkpoints'), note=f'best_val_{domain_name}_model')
            log_file.info(f'Model saved! Best Val Acc: {best_val*100:.2f}%')
        site_evaluation(i, args.test_domain, args, global_model, dataloader_dict[args.test_domain]['test'], log_file, log_ten, metric, note='test_domain')
        
    SaveCheckPoint(args, global_model, args.comm, os.path.join(log_dir, 'checkpoints'), note='last_model')
    for domain_name in dataobj.train_domain_list: # local model
        SaveCheckPoint(args, model_dict[domain_name], args.comm, os.path.join(log_dir, 'checkpoints'), note=f'last_{domain_name}_model')
    
    
if __name__ == '__main__':
    FLAGS(sys.argv, known_only=True)
    main()