## Distributted Parallel Training with SSL
# torch.nn.parallel.DistributedDataParallel is proven to be significantly faster 
# than torch.nn.DataParallel for single-node multi-GPU data parallel training.

import argparse
import collections
from cv2 import moments

import os, sys
sys.path.append('/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/Code/Yufeng/multimodal_CXR_Note')

import torch
import CXR_datasets.Dataloader_ddp as module_dataloader
import Models.DINO as module_arch
from utils.parse_config import ConfigParser
from trainer.trainer_DINO import DINO_Trainer
import utils.main_utils as module_utils
import utils.logger_utils as module_logger
import utils.augmentations as module_augmentation
import utils.ddp_utils as module_ddp_util
import torch.distributed as module_dist
import torch.nn as nn
import torch


def init_distributed_mode(args):
    # launched with torch.distributed.launch, some reference page for DDP:
    # https://pytorch.org/docs/stable/distributed.html#launch-utility
    # https://leimao.github.io/blog/PyTorch-Distributed-Training/#Launching-Distributed-Training
    # https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide
    # https://github.com/pytorch/examples/blob/main/distributed/ddp/README.md
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = "12355" # str(find_free_port())
    # launched with submit it on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    module_dist.init_process_group(
        backend="nccl",
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {})'.format(args.rank), flush=True)
    module_dist.barrier()

    module_ddp_util.setup_for_distributed_print(args.rank == 0)

    return args

def main(config, args):
    """
    The training parameter referenced the paper:
        Self-evolving vision transformer for chest X-ray diagnosis through knowledge distillation
    """

    # ########### Initialize the DDP process group ###########
    args = init_distributed_mode(args)
    local_rank = args.gpu
    torch.cuda.set_device(local_rank)
    # ########################################################

    # fix the random seed
    module_utils.fix_random_seeds()

    # get the rank of the current process
    run_rank = module_dist.get_rank()
    # obtain the logger instance with logger name and default log level
    # the logger would only be created on the rank 0 process
    logger = module_logger.get_logger('ssl_experiment', 
                                      config['trainer']['verbosity'],
                                      rank=run_rank)

    # setup the distributed data parallel dataloaders
    # In this dataloader, a customized DistributedSampler is used to makes 
    # sure that each process gets a different slice of the training data.
    data_loader = config.init_obj('data_loader', module_dataloader, 
                                  num_replicas=args.world_size)
    val_data_loader = data_loader.get_val_loader()

    # initialize the model architecture, then print to console
    model = config.init_obj('arch', module_arch)


    student, teacher, embed_dim = model.get_model()

    # load the pre-trained weights
    full_checkpoint = "/nfs/turbo/med-kayvan-lab/Projects/ARDS/Data/Processed/ARDS/CXR_Representation_Learning/Results/DINO/dino_deitsmall16_pretrain_full_checkpoint.pth"
    dino_16 = torch.load(full_checkpoint, map_location="cpu")
    student.load_state_dict({k.replace('module.', ''): v for k, v in dino_16['student'].items()}, strict=False)
    teacher.load_state_dict({k.replace('module.', ''): v for k, v in dino_16['teacher'].items()}, strict=False)
    
    student, teacher = student.cuda(), teacher.cuda()

    if module_ddp_util.is_main_process():
        logger.info('The model has been loaded.')

    # synchronize batch norms (if any)
    if module_ddp_util.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[local_rank], find_unused_parameters=True)
        teacher_without_ddp = teacher.module
        logger.info('The BatchNorm modules in the model has been converted to SyncBatchNorm.')
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher

    # wrap the model as a DistributedDataParallel model. 
    # this reproduces the model onto the GPU for the process.
    student = nn.parallel.DistributedDataParallel(student, device_ids=[local_rank], find_unused_parameters=True)
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    module_utils.set_requires_grad(teacher, False)

    # initialize the augmentation
    augmentation = config.init_obj('augmentation', module_augmentation).cuda()

    # build optimizer, learning rate scheduler.
    #      seperate the parameters into regularized and not regularized
    #      we do not regularize biases nor norm parameters
    params_groups = module_ddp_util.get_params_groups(student)
    
    # initialize the optimizer
    optimizer = config.init_obj('optimizer', torch.optim, params_groups)


    # get the number of epochs and the number of iterations per epoch
    epoch = config['trainer']['epochs']
    niter_per_ep = len(data_loader)

    # initialize all kinds of schedulers
    batch_size_per_gpu = config['data_loader']['args']['batch_size']
    config['lr_scheduler']['args']['base_value'] = config['lr_scheduler']['args']['base_value'] \
        * batch_size_per_gpu * args.world_size / 256.
    lr_scheduler = config.init_obj('lr_scheduler', module_utils, epochs=epoch,
                                    niter_per_ep=niter_per_ep)
    wd_scheduler = config.init_obj('wd_scheduler', module_utils, epochs=epoch,
                                    niter_per_ep=niter_per_ep)
    momentum_scheduler = config.init_obj('momentum_scheduler', module_utils, epochs=epoch,
                                    niter_per_ep=niter_per_ep)

    # initialize the loss
    # get the out_dim from model
    out_dim = config['arch']['args']['out_dim']
    # get the local_crops_number from augmentation
    local_crops_number = config['augmentation']['args']['local_crops_number']
    # get the epochs from trainer
    epochs = config['trainer']['epochs']

    dino_loss = config.init_obj('loss', module_arch, out_dim=out_dim,
                                ncrops=local_crops_number+2,
                                nepochs=epochs).cuda()

    # initialize the mix-precsion trainer
    fp16_scaler = torch.cuda.amp.GradScaler() if config['trainer']['use_fp16'] else None

    # resume training if needed
    to_restore = {"start_epoch": 1}
    if config.resume:
        module_utils.resume_from_checkpoint(config.resume, logger, 
                                            restore_variables=to_restore,
                                            student=student,
                                            teacher=teacher,
                                            optimizer=optimizer,
                                            fp16_scaler=fp16_scaler,
                                            dino_loss=dino_loss,)

    # Construct the trainer instance for the SSL network
    trainer = DINO_Trainer(config, student, teacher, teacher_without_ddp,
                            optimizer, dino_loss, augmentation, 
                            data_loader, val_data_loader, logger,
                            start_epoch=to_restore["start_epoch"], 
                            lr_scheduler=lr_scheduler,
                            wd_scheduler=wd_scheduler,
                            momentum_scheduler=momentum_scheduler,
                            fp16_scaler=fp16_scaler,
                            distributed=True)

    trainer.train()

    module_ddp_util.cleanup()

    
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Self Supervised Learning')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-k', '--local_rank', default=0, type=int,
                    help='ranking of the main node')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size') 
    ]
    # Get all the configurations from the config file and the command line arguments
    config = ConfigParser.from_args(module_utils.get_run_id, args, options)
    
    ddp_arg = args.parse_args()

    main(config, ddp_arg)