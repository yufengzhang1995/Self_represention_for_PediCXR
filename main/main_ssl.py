import argparse
import collections
import torch
import CXR_datasets.Dataloader as module_dataloader
# import byol as module_arch

import archive.DINO as module_arch
from utils.parse_config import ConfigParser
from trainer.trainer_ssl import BYOL_Trainer,DINO_Trainer
import utils.main_utils as module_utils
import utils.logger_utils as module_logger

def main(config):
    # fix the random seed
    module_utils.fix_random_seeds()

    # obtain the logger instance with logger name and default log level debug
    logger = module_logger.get_logger('ssl_experiment', 
                                      config['trainer']['verbosity'])

    # setup the dataloaders
    data_loader = config.init_obj('data_loader', module_dataloader)
    train_data_loader,val_data_loader = data_loader.get_data_loader()

    # initialize the model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info('The model has been loaded.')
    print(type(model))

    # prepare for (multi-device) GPU training
    device, device_ids = module_utils.prepare_device(config['n_gpu'])
    model = model.to(device)
    model.init_network(device)
    logger.info('The model has been initialized.')

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        logger.info('Running model on {} GPUs'.format(len(device_ids)))

    # build optimizer, learning rate scheduler.
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # - "How Transferable are Self-supervised Features in Medical Image Classification Tasks?"
    #   provided the hyper-param for the DINO model on Chest X-ray images.
    # - "Exploring Image Augmentations for Siamese Representation Learning with Chest X-Rays"
    #   provided the hyper-param for the BYOL model on Chest X-ray images.
        # The SGD optimizer (with weight decay= 0.0001 and momentum = 0.9) is used for 
        # pretraining (lr = 0.05), linear probing (lr = 30, weight decay= 0),
        # and fine-tuning (lr = 0.00001) using a cosine decay learning rate scheduler
    # - Self-defined. Using Adam optimizer (lr = 0.05) & LinearWarmupCosineAnnealingLR for pretraining

    optimizer, lr_scheduler = module_utils.build_optimizer_and_scheduler(config, 
                                                                         trainable_params)

    # resume training if needed
    to_restore = {"start_epoch": 1}
    if config.resume:
        module_utils.resume_from_checkpoint(config.resume, logger, 
                                            restore_variables=to_restore,
                                            model=model,
                                            optimizer=optimizer,
                                            lr_scheduler=lr_scheduler)

    # Construct the trainer instance for the SSL network
    if config["arch"]["type"] == "BYOL":
        trainer = BYOL_Trainer(config, model, optimizer, device, 
                            train_data_loader, val_data_loader, logger,
                            feature_func='features', 
                            start_epoch=to_restore["start_epoch"], 
                            lr_scheduler=lr_scheduler)
    elif config["arch"]["type"] == "DINO":
        trainer = DINO_Trainer(config, model, optimizer, device, 
                            train_data_loader, val_data_loader, logger,
                            feature_func='features', 
                            start_epoch=to_restore["start_epoch"], 
                            lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Self Supervised Learning')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size') 
    ]

    # Get all the configurations from the config file and the command line arguments
    config = ConfigParser.from_args(module_utils.get_run_id, args, options)

    main(config)