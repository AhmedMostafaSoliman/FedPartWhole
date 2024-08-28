#################################################################
########################### PACS ################################
#################################################################

#mobilenetv1 pretrained
export PYTHONPATH="$PYTHONPATH:." &&
cd FedPartWhole &&
python algorithms/fedprox/train_pacs.py --model "mobilenetv1" --pretrain --dataset pacs --test_domain p --num_classes 7 --comment "fedprox, mobilenetv1 pretrained, pacs on p" &&
python algorithms/fedprox/train_pacs.py --model "mobilenetv1" --pretrain --dataset pacs --test_domain a --num_classes 7 --comment "fedprox, mobilenetv1 pretrained, pacs on a" &&
python algorithms/fedprox/train_pacs.py --model "mobilenetv1" --pretrain --dataset pacs --test_domain c --num_classes 7 --comment "fedprox, mobilenetv1 pretrained, pacs on c" &&
python algorithms/fedprox/train_pacs.py --model "mobilenetv1" --pretrain --dataset pacs --test_domain s --num_classes 7 --comment "fedprox, mobilenetv1 pretrained, pacs on s"


#ccnet 3 heads
export PYTHONPATH="$PYTHONPATH:." &&
cd FedPartWhole &&
python algorithms/fedprox/train_pacs.py --flagfile configs/config_PACS_ccnet.cfg --model "ccnet" --dataset pacs --test_domain p --num_classes 7 --comment "fedprox, ccnet(3heads), pacs on p" &&
python algorithms/fedprox/train_pacs.py --flagfile configs/config_PACS_ccnet.cfg --model "ccnet" --dataset pacs --test_domain a --num_classes 7 --comment "fedprox, ccnet(3heads), pacs on a" &&
python algorithms/fedprox/train_pacs.py --flagfile configs/config_PACS_ccnet.cfg --model "ccnet" --dataset pacs --test_domain c --num_classes 7 --comment "fedprox, ccnet(3heads), pacs on c" &&
python algorithms/fedprox/train_pacs.py --flagfile configs/config_PACS_ccnet.cfg --model "ccnet" --dataset pacs --test_domain s --num_classes 7 --comment "fedprox, ccnet(3heads), pacs on s"


#################################################################
########################### VLCS ################################
#################################################################

#mobilenetv1 pretrained
export PYTHONPATH="$PYTHONPATH:." &&
cd FedPartWhole &&
python algorithms/fedprox/train_pacs.py --model "mobilenetv1" --pretrain --dataset vlcs --test_domain v --num_classes 5 --comment "fedprox, mobilenetv1 pretrained, vlcs on v" &&
python algorithms/fedprox/train_pacs.py --model "mobilenetv1" --pretrain --dataset vlcs --test_domain l --num_classes 5 --comment "fedprox, mobilenetv1 pretrained, vlcs on l" &&
python algorithms/fedprox/train_pacs.py --model "mobilenetv1" --pretrain --dataset vlcs --test_domain c --num_classes 5 --comment "fedprox, mobilenetv1 pretrained, vlcs on c" &&
python algorithms/fedprox/train_pacs.py --model "mobilenetv1" --pretrain --dataset vlcs --test_domain s --num_classes 5 --comment "fedprox, mobilenetv1 pretrained, vlcs on s"


#ccnet 3 heads
export PYTHONPATH="$PYTHONPATH:." &&
cd FedPartWhole &&
python algorithms/fedprox/train_pacs.py --flagfile configs/config_VLCS_ccnet.cfg --model "ccnet" --dataset vlcs --test_domain v --num_classes 5 --comment "fedprox, ccnet(3heads), vlcs on v" &&
python algorithms/fedprox/train_pacs.py --flagfile configs/config_VLCS_ccnet.cfg --model "ccnet" --dataset vlcs --test_domain l --num_classes 5 --comment "fedprox, ccnet(3heads), vlcs on l" &&
python algorithms/fedprox/train_pacs.py --flagfile configs/config_VLCS_ccnet.cfg --model "ccnet" --dataset vlcs --test_domain c --num_classes 5 --comment "fedprox, ccnet(3heads), vlcs on c" &&
python algorithms/fedprox/train_pacs.py --flagfile configs/config_VLCS_ccnet.cfg --model "ccnet" --dataset vlcs --test_domain s --num_classes 5 --comment "fedprox, ccnet(3heads), vlcs on s"
