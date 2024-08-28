from network import ResNet, CCNet, Agglomerator
import torch
import timm
from transformers import MobileNetV1ForImageClassification, MobileNetV1Config


def GetNetwork(args, num_classes, pretrained=False, **kwargs):
    if args.model == 'resnet18':
        model = ResNet.resnet18(pretrained=pretrained, num_classes=num_classes, **kwargs)
        feature_level = 512
        
    elif args.model == 'resnet18_rsc':
        model = ResNet.resnet18_rsc(pretrained=pretrained, num_classes=num_classes, **kwargs)
        feature_level = 512

    elif args.model == 'resnet50':
        model = ResNet.resnet50(pretrained=pretrained, num_classes=num_classes, **kwargs)
        feature_level = 2048
        
    elif args.model == 'resnet50_rsc':
        model = ResNet.resnet50_rsc(pretrained=pretrained, num_classes=num_classes, **kwargs)
        feature_level = 2048
    
    elif args.model == 'mobilenetv2':
        model = timm.create_model('mobilenetv2_120d.ra_in1k', pretrained=True, num_classes=num_classes)
        feature_level = 1280
    
    elif args.model == 'mobilenetv1':
        class CustomMobileNetV1ModelForImageClassification(MobileNetV1ForImageClassification):
            def forward(self, pixel_values, **kwargs):
                outputs = super().forward(pixel_values, **kwargs)
                return outputs.logits  # Return only the logits

        conf = MobileNetV1Config(name_or_path='google/mobilenet_v1_1.0_224', num_labels=num_classes)
        model = CustomMobileNetV1ModelForImageClassification.from_pretrained("google/mobilenet_v1_1.0_224", ignore_mismatched_sizes=False)
        print(model)
        feature_level = 1024

    elif args.model == 'agg':
        model = Agglomerator.Agglomerator(args.FLAGS)
        if args.agg_ckpt is not None:
            checkpoint = torch.load(args.agg_ckpt)
            new_state_dict = {k: v for k, v in checkpoint['model'].items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
            model.load_state_dict(new_state_dict, strict=False)
            print('agg model loaded')
        feature_level = 256
    
    elif args.model == 'ccnet':
        model = CCNet.CCNet(args.FLAGS)
        feature_level = 256

    else:
        raise ValueError("The model is not supported")
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total params:", pytorch_total_params)

    return model, feature_level
