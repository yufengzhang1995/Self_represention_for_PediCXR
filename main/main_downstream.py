import argparse
import collections
import importlib
import torch
import CXR_datasets.Dataloader as module_dataloader
import Models.model_downstream as module_arch
import torchvision.models as module_model
import torch.optim as module_optimizer
import torch.optim.lr_scheduler as module_scheduler
from utils.parse_config import ConfigParser
import utils.main_utils as module_utils
import utils.logger_utils as module_logger
import utils.augmentations as module_augment
import os

def main(config, fold=-1):
    # fix the random seed
    module_utils.fix_random_seeds()

    # obtain the logger instance with logger name and default log level debug
    logger = module_logger.get_logger('tram_downstream', 
                                      config['trainer']['verbosity'])
    # prepare for training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setup the dataloaders
    if "KFold" in config['data_loader']['type']:
        assert fold >= 0, "The fold number should be specified for 'KFold' data loader."
        data_loader_obj = config.init_obj('data_loader', module_dataloader)
        data_loader, val_data_loader = data_loader_obj.get_data_loaders(fold)
    else:   
        data_loader_obj = None
        data_loader = config.init_obj('data_loader', module_dataloader)
        val_data_loader = data_loader.get_val_loader()

    # Prepare for model intialization
    data_dim_add = data_loader.get_additional_dim() if data_loader_obj is \
        None else data_loader_obj.get_additional_dim()

    feature_num_mapping = {"torchxrayvision.autoencoders.ResNetAE":4608, 
                        "torchxrayvision.models.ResNet":2048, 
                        "torchxrayvision.models.DenseNet":1024,
                        "torchvision.models.resnet50":2048,
                        "transformer.vit_small":384}           
    feature_func_mapping = {"torchxrayvision.autoencoders.ResNetAE":'features',
                        "torchxrayvision.models.ResNet":"features",
                        "torchxrayvision.models.DenseNet":"features2",
                        "torchvision.models.resnet50":None,
                        "transformer.vit_small":None}
    
    encoder_type = config['encoder']['type']
    encoder_args = config['encoder']['args']

    if encoder_type in feature_num_mapping:
        feature_num = feature_num_mapping[encoder_type]
        feature_func = feature_func_mapping[encoder_type]
    else:
        raise ValueError("The encoder type is not supported.")
    
    if encoder_type.startswith('torchxrayvision.'):
        module_name = '.'.join(encoder_type.split('.')[:-1])
        encoder_name = encoder_type.split('.')[-1]
        module_encoder= importlib.import_module(module_name)
        encoder = getattr(module_encoder, encoder_name)(
            weights=encoder_args['weights'])
        encoder = encoder.to(device)

        if encoder_args['freeze']:
            # freeze the encoder if specified
            module_utils.set_requires_grad(encoder, False)

    elif encoder_type.startswith('transformer.'):
        import Model.model_transformer as module_transformer
        encoder = getattr(module_transformer, encoder_type.split('.')[-1])(
            patch_size=encoder_args['patch_size'],
            img_size=config['data_loader']['args']['image_size'],
            drop_path_rate=encoder_args['drop_path_rate'])
        if encoder_args['weights'] == 'imagenet':
            loaded_weight = torch.hub.load('facebookresearch/dino:main', 
                                           f'dino_vits{encoder_args["patch_size"]}')
            state_dict = loaded_weight.state_dict()
        elif os.path.isfile(encoder_args['weights']):
            loaded_weight = torch.load(encoder_args['weights'], map_location='cpu')
            state_dict = {k.replace('dino.', ''): v for k, v in loaded_weight[
                'state_dict'].items() if not k.startswith('linear')}
        else:
            raise ValueError("The encoder weights is not supported.")
        
        encoder.load_state_dict(state_dict, strict=False)
        encoder = encoder.to(device)

        if encoder_args['freeze']:
            # freeze the encoder if specified
            module_utils.set_requires_grad(encoder, False)
            
    elif encoder_type.startswith('torchvision.'):
        layer_drop = -1 # default drop the last layer
        if encoder_args['weights'] == 'imagenet':
            encoder = getattr(module_model, encoder_type.split('.')[-1])(pretrained=True)
        elif encoder_args['weights'] == 'random':
            encoder = getattr(module_model, encoder_type.split('.')[-1])(pretrained=False)
        else:
            encoder = getattr(module_model, encoder_type.split('.')[-1])(pretrained=False)
        
        if 'in_channels' in encoder_args:
            in_channels = encoder_args['in_channels']
            if in_channels == 1 and encoder.__class__.__name__ == 'ResNet':
                updated_conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                torch.nn.init.normal_(updated_conv1.weight, std=0.01)  
                encoder.conv1 = updated_conv1

        # load the previous trained weights if end with .pth
        if encoder_args['weights'].endswith('.pth'):
            loaded_weight = torch.load(encoder_args['weights'], map_location='cpu')
            if 'backbone' in loaded_weight:
                encoder.load_state_dict(loaded_weight['backbone'])
                assert feature_num == loaded_weight[
                    'feature_dim'], "The feature dimension is not matched."
                layer_drop = loaded_weight['layer_drop']
                assert loaded_weight['image_size']==config['data_loader'][
                    'args']['image_size'], "The image size is not matched."
            else: # It's an undefined model 
                encoder.load_state_dict(loaded_weight)

        if encoder_args['freeze']:
            # freeze the encoder if specified
            module_utils.set_requires_grad(encoder, False)
            if encoder_args['weights'].endswith('.pth') and config['arch']['type'].startswith('TRAM')\
                and encoder.__class__.__name__ == 'ResNet':
                module_utils.set_requires_grad(encoder.layer4[-1], True) #[-1]
                logger.info("The encoder is frozen except for the final functional block")

        # drop the last layer
        encoder = torch.nn.Sequential(*list(encoder.children())[:layer_drop])
        encoder = encoder.to(device)

    # initialize the model architecture, then print to console
    arch_type = config['arch']['type']

    if arch_type.startswith('TRAM'):
        if not encoder_args['freeze']:
            logger.info("The encoder is not frozen.")
        model = config.init_obj('arch', module_arch, encoder, 
                                add_input_dim=data_dim_add,
                                feature_function=feature_func,
                                base_predictor_input_dim=feature_num)
    elif arch_type == 'Linear_Eval':
        if encoder_args['freeze']:
            logger.info("The encoder is frozen for linear probing.")
        else:
            logger.info("The encoder is not frozen for fine-tuning.")
        # implement the linear probing model
        model = config.init_obj('arch', module_arch, encoder,
                                input_dim=feature_num,
                                feature_function=feature_func)
    elif arch_type == 'Vits_Linear_Eval':
        model = config.init_obj('arch', module_arch, encoder,
                                input_dim=feature_num,
                                feature_function=feature_func)
    else:
        raise ValueError(f"The architecture type {arch_type} is not supported.")
    
    traineble_param = sum(p.numel() for p in model.parameters() if p.requires_grad)/(1e6)
    
    model = model.to(device)
    logger.info('The model has been loaded.')
    logger.info(f"The total number of trainable parameters is {traineble_param :.2f}M")

    device, device_ids = module_utils.prepare_device(config['n_gpu'])
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        logger.info('Running model on {} GPUs with DataParallel'.format(len(device_ids)))

    # separate the parameters into two groups
    model_parameters, encoder_parameters = [], []
    
    for name, param in model.named_parameters():
        if name.startswith('encoder'): 
            # Contain either the encoder parameters in linear eval model
            # or the base branch (only encoder) in TRAM model
            encoder_parameters.append(param)
        else: 
            # Contain either the parameters in the TRAM priviledged branch and the base predictor
            # or the parameters of classifiers in the linear eval model
            model_parameters.append(param)

    if arch_type == 'Linear_Eval' and encoder_args['freeze']:
        # build optimizer, learning rate scheduler.
        trainable_params = filter(lambda p: p.requires_grad, model_parameters)
        # initialize the optimizer and lr_scheduler
        optimizer = config.init_obj('optimizer', module_optimizer, trainable_params)
    else:  # run with fine-tune or TRAM model
        assert "lr_encoder" in encoder_args and "lr" in encoder_args, \
            "lr_encoder or lr is not specified in the config file."
        assert "lr" not in config['optimizer']['args'], \
            "lr should be specified in the encoder config."
        # 1. in the tram model, lr_encoder is the learning rate for the base branch
        # and lr is the learning rate for the priviledged branch and base branch predictor
        # 2. in the linear eval model, lr_encoder is the learning rate for the encoder
        # and lr is the learning rate for the classifier
        model_train_parameters = filter(lambda p: p.requires_grad, model_parameters)
        encoder_train_parameters = filter(lambda p: p.requires_grad, encoder_parameters)
        param_groups = [dict(params=model_train_parameters, lr=encoder_args['lr']),
                        dict(params=encoder_train_parameters, lr=encoder_args['lr_encoder'])]
        
        # Include the case for confusion matrix based training
        if config['trainer']['type'] == 'Confusion_Trainer':
            from utils.confusion_utils import confusion_matrix_estimators
            # initialize the confusion matrix parameters
            confusion_matrix = confusion_matrix_estimators(
                num_annotators=data_loader.dataset.get_num_reviewer(),
                num_classes=data_loader.dataset.get_num_label()).to(device)
            # set requires_grad=True for the confusion matrix parameters
            confusion_matrix.requires_grad = True
            # assign a smaller learning rate for the confusion matrix
            param_groups.append(dict(params=[confusion_matrix], lr=encoder_args['lr_encoder']))
        
        optimizer = config.init_obj('optimizer', module_optimizer, param_groups)

    # build learning rate scheduler
    lr_scheduler = config.init_obj(
        'lr_scheduler', module_scheduler, optimizer) if \
            'lr_scheduler' in config.config else None
    
    augmentation = config.init_obj(
        'augmentation', module_augment, config['data_loader']['args']['image_size']) if \
            'augmentation' in config.config else None

    # Construct the trainer instance
    if config['trainer']['type'] == 'Confusion_Trainer':
        import trainer_downstream as module_trainer
        trainer = getattr(module_trainer, config['trainer']['type'])(
                            confusion_matrix, config, 
                            model, optimizer, logger, device, 
                            data_loader, val_data_loader, 
                            augmentation, lr_scheduler, fold=fold)
    else:
        import trainer_downstream as module_trainer
        trainer = getattr(module_trainer, config['trainer']['type'])(config, 
                                model, optimizer, logger, device, 
                                data_loader, val_data_loader, 
                                augmentation, lr_scheduler, fold=fold)
    
    test_data_loader = data_loader_obj.get_test_loader() if \
        data_loader_obj is not None else data_loader.get_test_loader()
    
    if not config.test_only:
        trainer.train()
        
    trainer.test(test_data_loader)
    

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Self Supervised Learning')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-t', '--test_only', default=False, type=str,
                      help='if only the test are run (default: False)')
    
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--en', '--encoding'], type=str, target='data_loader;args;uncertainty_encoding'),
        CustomArgs(['--fc', '--decision_function'], type=int, target='data_loader;args;use_uncertainty_func'),
        CustomArgs(['--th', '--threshold'], type=float, target='data_loader;args;uncertainty_threshold'),
        CustomArgs(['--elr', '--encoder_learning_rate'], type=float, target='encoder;args;lr_encoder'),  
        CustomArgs(['--mlr', '--model_learning_rate'], type=float, target='encoder;args;lr'),
        CustomArgs(['--lf', '--loss_factor'], type=float, target='loss;args;additon_loss_factor'),
        CustomArgs(['--n', '--task_name'], type=str, target='name'),
        CustomArgs(['--wt', '--weight'], type=str, target='encoder;args;weights'),
        CustomArgs(['--cl', '--clean_data'], type=int, target='data_loader;args;clean_data_only'),
        CustomArgs(['--tn', '--trainer'], type=str, target='trainer;type'),
        CustomArgs(['--ep', '--epoch'], type=int, target='trainer;epochs'),
        CustomArgs(['--arc', '--architecture'], type=str, target='arch;type'),
    ]

    # Get all the configurations from the config file and the command line arguments
    config = ConfigParser.from_args(module_utils.get_run_id, args, options)

    # Perform cross validation if specified in the config file
    folds = config["cross_validation"] if "cross_validation" in config.config else -1

    if folds == -1:
        main(config)
    else:
        assert "KFold" in config['data_loader']['type'], \
            "Cross validation is only supported for 'KFold' data loader."
        config['data_loader']['args']['folds'] = folds
        for fold in range(folds):
            main(config, fold=fold)