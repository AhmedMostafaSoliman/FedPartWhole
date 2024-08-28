#resnet18 pretrained and not pretrained
export PYTHONPATH="$PYTHONPATH:." &&
cd FedPartWhole &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS.cfg --model "resnet18" --no-pretrain --test_domain p --comment "fed avg, resnet18 not pre, pacs on p, " &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS.cfg --model "resnet18" --no-pretrain --test_domain a --comment "fed avg, resnet18 not pre, pacs on a" &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS.cfg --model "resnet18" --no-pretrain --test_domain c --comment "fed avg, resnet18 not pre, pacs on c" &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS.cfg --model "resnet18" --no-pretrain --test_domain s --comment "fed avg, resnet18 not pre, pacs on s" &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS.cfg --model "resnet18" --pretrain --test_domain p --comment "fed avg, resnet18 pretrained, pacs on p" &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS.cfg --model "resnet18" --pretrain --test_domain a --comment "fed avg, resnet18 pretrained, pacs on a" &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS.cfg --model "resnet18" --pretrain --test_domain c --comment "fed avg, resnet18 pretrained, pacs on c" &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS.cfg --model "resnet18" --pretrain --test_domain s --comment "fed avg, resnet18 pretrained, pacs on s"


#mobilenetv2 pretrained
export PYTHONPATH="$PYTHONPATH:." &&
cd FedPartWhole &&
python algorithms/fedavg/train_pacs.py --model "mobilenetv2" --pretrain --test_domain p --comment "fed avg, mobilenetv2 pretrained, pacs on p" &&
python algorithms/fedavg/train_pacs.py --model "mobilenetv2" --pretrain --test_domain a --comment "fed avg, mobilenetv2 pretrained, pacs on a" &&
python algorithms/fedavg/train_pacs.py --model "mobilenetv2" --pretrain --test_domain c --comment "fed avg, mobilenetv2 pretrained, pacs on c" &&
python algorithms/fedavg/train_pacs.py --model "mobilenetv2" --pretrain --test_domain s --comment "fed avg, mobilenetv2 pretrained, pacs on s"

#mobilenetv1 pretrained
export PYTHONPATH="$PYTHONPATH:." &&
cd FedPartWhole &&
python algorithms/fedavg/train_pacs.py --model "mobilenetv1" --pretrain --test_domain p --comment "fed avg, mobilenetv1 pretrained, pacs on p" &&
python algorithms/fedavg/train_pacs.py --model "mobilenetv1" --pretrain --test_domain a --comment "fed avg, mobilenetv1 pretrained, pacs on a" &&
python algorithms/fedavg/train_pacs.py --model "mobilenetv1" --pretrain --test_domain c --comment "fed avg, mobilenetv1 pretrained, pacs on c" &&
python algorithms/fedavg/train_pacs.py --model "mobilenetv1" --pretrain --test_domain s --comment "fed avg, mobilenetv1 pretrained, pacs on s"


#ccnet 3 heads
export PYTHONPATH="$PYTHONPATH:." &&
cd FedPartWhole &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS_ccnet.cfg --model "ccnet" --test_domain p --comment "fed avg, ccnet(3heads), pacs on p" &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS_ccnet.cfg --model "ccnet" --test_domain a --comment "fed avg, ccnet(3heads), pacs on a" &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS_ccnet.cfg --model "ccnet" --test_domain c --comment "fed avg, ccnet(3heads), pacs on c" &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS_ccnet.cfg --model "ccnet" --test_domain s --comment "fed avg, ccnet(3heads), pacs on s"


#ccnet 2 heads
export PYTHONPATH="$PYTHONPATH:." &&
cd FedPartWhole &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS_ccnet.cfg --model "ccnet" --test_domain p --comment "fed avg, ccnet(2heads), pacs on p" &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS_ccnet.cfg --model "ccnet" --test_domain a --comment "fed avg, ccnet(2heads), pacs on a" &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS_ccnet.cfg --model "ccnet" --test_domain c --comment "fed avg, ccnet(2heads), pacs on c" &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS_ccnet.cfg --model "ccnet" --test_domain s --comment "fed avg, ccnet(2heads), pacs on s"

#ccnet 1 heads
export PYTHONPATH="$PYTHONPATH:." &&
cd FedPartWhole &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS_ccnet.cfg --model "ccnet" --test_domain p --comment "fed avg, ccnet(1 head), pacs on p" &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS_ccnet.cfg --model "ccnet" --test_domain a --comment "fed avg, ccnet(1 head), pacs on a" &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS_ccnet.cfg --model "ccnet" --test_domain c --comment "fed avg, ccnet(1 head), pacs on c" &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS_ccnet.cfg --model "ccnet" --test_domain s --comment "fed avg, ccnet(1 head), pacs on s"


#Distill partwhole pretrain
export PYTHONPATH="$PYTHONPATH:." &&
cd FedPartWhole &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS_pretrain.cfg --model "agg" --test_domain p --comment "fed avg, agglomerator(pretrainstep), pacs on p" &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS_pretrain.cfg --model "agg" --test_domain a --comment "fed avg, agglomerator(pretrainstep), pacs on a" &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS_pretrain.cfg --model "agg" --test_domain c --comment "fed avg, agglomerator(pretrainstep), pacs on c" &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS_pretrain.cfg --model "agg" --test_domain s --comment "fed avg, agglomerator(pretrainstep), pacs on s"


#Distill Part Whole finetune from pretrained
export PYTHONPATH="$PYTHONPATH:." &&
cd FedPartWhole &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS_finetune.cfg --model "agg" --agg_ckpt="exps/pacs_fedavg/distill_part_whole/pre/exp190-2024-03-16-17-17-fedavg_train_pacs-pacs-agg-0.0003-bs256-comm40-generalization_adjustment/checkpoints/best_val_model.pt" --test_domain p --comment "fed avg, agglomerator(finetunestep fixed), pacs on p" &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS_finetune.cfg --model "agg" --agg_ckpt="exps/pacs_fedavg/distill_part_whole/pre/exp191-2024-03-16-19-36-fedavg_train_pacs-pacs-agg-0.0003-bs256-comm40-generalization_adjustment/checkpoints/best_val_model.pt" --test_domain a --comment "fed avg, agglomerator(finetunestep fixed), pacs on a" &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS_finetune.cfg --model "agg" --agg_ckpt="exps/pacs_fedavg/distill_part_whole/pre/exp192-2024-03-16-21-48-fedavg_train_pacs-pacs-agg-0.0003-bs256-comm40-generalization_adjustment/checkpoints/best_val_model.pt" --test_domain c --comment "fed avg, agglomerator(finetunestep fixed), pacs on c" &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS_finetune.cfg --model "agg" --agg_ckpt="exps/pacs_fedavg/distill_part_whole/pre/exp193-2024-03-16-23-57-fedavg_train_pacs-pacs-agg-0.0003-bs256-comm40-generalization_adjustment/checkpoints/best_val_model.pt" --test_domain s --comment "fed avg, agglomerator(finetunestep fixed), pacs on s"

#Agglomerator finetune from scratch
export PYTHONPATH="$PYTHONPATH:." &&
cd FedPartWhole &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS_finetune.cfg --model "agg" --agg_ckpt="" --test_domain p --comment "fed avg, agglomerator(no-pre), pacs on p" &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS_finetune.cfg --model "agg" --agg_ckpt="" --test_domain a --comment "fed avg, agglomerator(no-pre), pacs on a" &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS_finetune.cfg --model "agg" --agg_ckpt="" --test_domain c --comment "fed avg, agglomerator(no-pre), pacs on c" &&
python algorithms/fedavg/train_pacs.py --flagfile configs/config_PACS_finetune.cfg --model "agg" --agg_ckpt="" --test_domain s --comment "fed avg, agglomerator(no-pre), pacs on s"