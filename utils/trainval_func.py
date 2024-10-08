import os
from network.get_network import GetNetwork
import torch
from configs.default import *
import torch.nn.functional as F
from tqdm import tqdm
import random
from utils.ccutils import levels_recon_loss_3lvls, plot_islands_agreement
import wandb
from lion_pytorch import Lion
import argparse
import matplotlib.pyplot as plt

site_to_dir_name_dict = {
    'p': 'photo',
    'a': 'art_painting',
    'c': 'cartoon',
    's': 'sketch'
}

def Shuffle_Batch_Data(data_in):
    len_total = len(data_in)
    idx_list = list(range(len_total))
    random.shuffle(idx_list)
    return data_in[idx_list]

def epoch_site_train(epochs, site_name, model, optimzier, scheduler, dataloader, log_ten, metric):
    model.train()
    for i, data_list in enumerate(dataloader):
        #if model.__class__.__name__ == 'Agglomerator':
        imgs, labels, domain_labels, embeds, indices = data_list
        embeds = embeds.cuda()
        #else:
        #    imgs, labels, domain_labels = data_list
        imgs = imgs.cuda()
        labels = labels.cuda()
        domain_labels = domain_labels.cuda()
        optimzier.zero_grad()
        if model.__class__.__name__ == 'CCNet':
            output, _ = model(imgs, levels=embeds) #glom
        elif model.__class__.__name__ == 'Agglomerator':
            output, _ = model(imgs)
        else:
            output = model(imgs)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        optimzier.step()
        log_ten.add_scalar(f'{site_name}_train_loss', loss.item(), epochs*len(dataloader)+i)
        metric.update(output, labels)
    
    log_ten.add_scalar(f'{site_name}_train_acc', metric.results()['acc'], epochs)
    scheduler.step()

def epoch_site_pretrain_agg(epochs, site_name, model, optimzier, scheduler, dataloader, log_ten, metric):
    model.train()
    for i, data_list in enumerate(dataloader):
        imgs, labels, domain_labels,  embeds, indices = data_list
        imgs = imgs.cuda()
        labels = labels.cuda()
        embeds = embeds.cuda()
        domain_labels = domain_labels.cuda()
        optimzier.zero_grad()
        top_level, toplot, encoded_levels_out = model(imgs)
        loss = levels_recon_loss_3lvls(encoded_levels_out, embeds)
        loss.backward()
        optimzier.step()
        log_ten.add_scalar(f'{site_name}_train_loss', loss.item(), epochs*len(dataloader)+i)
    
    scheduler.step()

    
def site_train(comm_rounds, site_name, args, model, optimizer, scheduler, dataloader, log_ten, metric):
    tbar = tqdm(range(args.local_epochs))
    for local_epoch in tbar:
        tbar.set_description(f'{site_name}_train')
        if args.model == 'agg' and args.FLAGS.supervise == False:
            epoch_site_pretrain_agg(comm_rounds*args.local_epochs + local_epoch, site_name, model, optimizer, scheduler, dataloader, log_ten, metric)
        else:
            epoch_site_train(comm_rounds*args.local_epochs + local_epoch, site_name, model, optimizer, scheduler, dataloader, log_ten, metric)
            
def site_evaluation(epochs, site_name, args, model, dataloader, log_file, log_ten, metric, note='after_fed'):
    model.eval()
    with torch.no_grad():
        for items in dataloader:
            imgs = items[0]
            labels = items[1]
            domain_labels = items[2]
            imgs = imgs.cuda()
            n_images =  len(imgs)
            if args.model == 'agg' or args.model == 'ccnet':
                embeds = items[3]
                embeds = embeds.cuda()
                if args.model == 'ccnet':
                    output, encoded_levels_out = model(imgs, levels=embeds) #glom
                else:
                    output, encoded_levels_out = model(imgs)
                #plot a random image from the batch
                if epochs % 2 == 0:
                    idx = random.randint(0, n_images-1)
                    fig1 = plot_islands_agreement(encoded_levels_out[idx,:,:,:], imgs[idx,:,:,:])
                    fig2 = plot_islands_agreement(embeds[idx,:,:,:], imgs[idx,:,:,:])
                    wandb.log({"Learned Islands": fig1, "GT EMbeddings": fig2})
                    #close fig1 and fig2
                    plt.close(fig1)
                    plt.close(fig2)
            else:
                output = model(imgs)
            metric.update(output, labels)
    results_dict = metric.results()
    log_ten.add_scalar(f'{note}_{site_name}_loss', results_dict['loss'], epochs)
    log_ten.add_scalar(f'{note}_{site_name}_acc', results_dict['acc'], epochs)
    log_file.info(f'{note} Round: {epochs:3d} | Epochs: {args.local_epochs*epochs:3d} | Domain: {site_name} | loss: {results_dict["loss"]:.4f} | Acc: {results_dict["acc"]*100:.2f}%')

    return results_dict

def site_pre_evaluation_agg(epochs, site_name, args, model, dataloader, log_file, log_ten, metric, note='after_fed'):
    model.eval()
    loss = 0
    total_count = 0
    with torch.no_grad():
        for imgs, labels, domain_labels, embeds, indices in dataloader:
            rootdir = f'datasets/{args.dataset.upper()}/raw_images/{site_to_dir_name_dict[site_name]}'
            paths = []
            # get image_paths from the indices
            
            for cls_idx, sample_idx in zip(labels, indices):
                class_path = os.path.join(rootdir, sorted(os.listdir(rootdir))[cls_idx])
                #print('files len', len(os.listdir(class_path)))
                ##print('sample_idx', sample_idx)
                #paths.append(sorted(os.listdir(class_path))[sample_idx])

            imgs = imgs.cuda()
            embeds = embeds.cuda()
            top_level, toplot, encoded_levels_out = model(imgs)
            loss += levels_recon_loss_3lvls(encoded_levels_out, embeds)
            total_count += len(labels)
            if epochs % 2 == 0:
                fig1 = plot_islands_agreement(toplot, imgs[0,:,:,:])
                fig2 = plot_islands_agreement(embeds[0,:,:,:], imgs[0,:,:,:])
                # split the string and get the name of the image from the whole path
                #fig1.suptitle(paths[0])
                #fig2.suptitle(paths[0])
                wandb.log({"Learned Islands": fig1, "GT EMbeddings": fig2})


    #results_dict = metric.results()
    results_dict =  {'loss':float(loss) / float(total_count)}
    log_ten.add_scalar(f'{note}_{site_name}_loss', float(loss) / float(total_count), epochs)
    log_file.info(f'{note} Round: {epochs:3d} | Epochs: {args.local_epochs*epochs:3d} | Domain: {site_name} | loss: {results_dict["loss"]:.4f}')

    return results_dict

def site_evaluation_class_level(epochs, site_name, args, model, dataloader, log_file, log_ten, metric, note='after_fed'):
    model.eval()
    with torch.no_grad():
        for imgs, labels, domain_labels, in dataloader:
            imgs = imgs.cuda()
            output = model(imgs)
            metric.update(output, labels)
    results_dict = metric.results()
    log_ten.add_scalar(f'{note}_{site_name}_loss', results_dict['loss'], epochs)
    log_ten.add_scalar(f'{note}_{site_name}_acc', results_dict['acc'], epochs)
    log_ten.add_scalar(f'{note}_{site_name}_class_acc', results_dict['class_level_acc'], epochs)
    log_file.info(f'{note} Round: {epochs:3d} | Epochs: {args.local_epochs*epochs:3d} | Domain: {site_name} | loss: {results_dict["loss"]:.4f} | Acc: {results_dict["acc"]*100:.2f}% | C Acc: {results_dict["class_level_acc"]*100:.2f}%')

    return results_dict

def site_only_evaluation(model, dataloader, metric):
    model.eval()
    with torch.no_grad():
        for imgs, labels, domain_labels, in dataloader:
            imgs = imgs.cuda()
            output = model(imgs)
            metric.update(output, labels)
    results_dict = metric.results()
    return results_dict

def GetFedModel(args, num_classes, is_train=True):
    global_model, feature_level = GetNetwork(args, args.num_classes, is_train)
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
        #optimizer_dict[domain_name] = torch.optim.SGD(model_dict[domain_name].parameters(), lr=args.lr, momentum=0.9,
        #                                              weight_decay=5e-4)
        optimizer_dict[domain_name] = Lion(model_dict[domain_name].parameters(),
                                           lr=args.lr,
                                           betas= (0.95, 0.98),
                                           weight_decay=0.05)

        total_epochs = args.local_epochs * args.comm
        if args.lr_policy == 'step':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.StepLR(optimizer_dict[domain_name], step_size=int(total_epochs *0.8), gamma=0.1)
        elif args.lr_policy == 'mul_step':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.MultiStepLR(optimizer_dict[domain_name], milestones=[int(total_epochs*0.3), int(total_epochs*0.8)], gamma=0.1)
        elif args.lr_policy == 'exp95':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.ExponentialLR(optimizer_dict[domain_name], gamma=0.95)
        elif args.lr_policy == 'exp98':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.ExponentialLR(optimizer_dict[domain_name], gamma=0.98)
        elif args.lr_policy == 'exp99':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.ExponentialLR(optimizer_dict[domain_name], gamma=0.99)   
        elif args.lr_policy == 'cos':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_dict[domain_name], T_max=total_epochs)
        elif args.lr_policy == 'one_cycle':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.OneCycleLR(
                optimizer_dict[domain_name],
                max_lr=args.lr,
                epochs=total_epochs,
                steps_per_epoch=45000//args.batch_size)
            
    return global_model, model_dict, optimizer_dict, scheduler_dict

def SaveCheckPoint(args, model, epochs, path, optimizer=None, schedule=None, note='best_val'):
    #remove flags from args
    args_copy = argparse.Namespace(**vars(args))
    if 'FLAGS' in vars(args_copy):
        delattr(args_copy, 'FLAGS')
    check_dict = {'args':args_copy, 'epochs':epochs, 'model':model.state_dict(), 'note': note}
    if optimizer is not None:
        check_dict['optimizer'] = optimizer.state_dict()
    if schedule is not None:
        check_dict['shceduler'] = schedule.state_dict()
    if not os.path.isdir(path):
        os.makedirs(path)
        
    torch.save(check_dict, os.path.join(path, note+'.pt'))
    

