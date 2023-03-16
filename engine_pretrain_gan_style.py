import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


# Utility functions
def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)
    


def train_one_epoch(model_g: torch.nn.Module,
                    model_d: torch.nn.Module,
                    data_loader: Iterable, 
                    optimizer_g: torch.optim.Optimizer,
                    optimizer_d: torch.optim.Optimizer,
                    loss_d: torch.nn.BCELoss,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    # accum_iter = args.accum_iter


    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, gray_samples) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer_g, data_iter_step / len(data_loader) + epoch, args)
        lr_sched.adjust_learning_rate(optimizer_d, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        gray_samples = gray_samples.to(device, non_blocking=True)
        
        # update generator
        optimizer_g.zero_grad()
        with torch.cuda.amp.autocast():
            toggle_grad(model_g, True)
            toggle_grad(model_d, False)
            model_g.train()
            model_d.eval()
            loss_reconstruction, pred, _ = model_g(gray_samples, samples, mask_ratio=args.mask_ratio)
            pred = model_g.unpatchify(pred)
            
            d_score = model_d(pred)
            loss_classify = loss_d(d_score, 1)
            loss_g = loss_reconstruction + loss_classify
            
            # modify
            # loss, _, _ = model(samples, samples, mask_ratio=args.mask_ratio)

        loss_g_value = loss_g.item()

        if not math.isfinite(loss_g_value):
            print("Loss g step is {}, stopping training".format(loss_g_value))
            sys.exit(1)

        loss_scaler(loss_g, optimizer_g, parameters=model_g.parameters(),
                    update_grad=True)
        

        torch.cuda.synchronize()

        metric_logger.update(loss_g=loss_g_value)

        lr_g = optimizer_g.param_groups[0]["lr"]
        metric_logger.update(lr_g=lr_g)

        loss_g_value_reduce = misc.all_reduce_mean(loss_g_value)
        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_generator_loss', loss_g_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr_g', lr_g, epoch_1000x)

        # update discriminator
        optimizer_d.zero_grad()
        with torch.cuda.amp.autocast():
            toggle_grad(model_g, False)
            toggle_grad(model_d, True)
            model_g.eval()
            model_d.train()
            
            with torch.no_grad():
                loss_reconstruction, pred, _ = model_g(gray_samples, samples, mask_ratio=args.mask_ratio)
                pred = model_g.unpatchify(pred)
            pred = pred.detach()
            d_score_fake = model_d(pred)
            loss_classify_fake = loss_d(d_score_fake, 0)
            d_score_real = model_d(samples)
            loss_classify_real = loss_d(d_score_real, 1)
            
            loss_d = loss_classify_fake + loss_classify_real
            
            # modify
            # loss, _, _ = model(samples, samples, mask_ratio=args.mask_ratio)

        loss_d_value = loss_d.item()

        if not math.isfinite(loss_d_value):
            print("Loss g step is {}, stopping training".format(loss_d_value))
            sys.exit(1)

        loss_scaler(loss_d, optimizer_d, parameters=model_d.parameters(),
                    update_grad=True)

        torch.cuda.synchronize()

        metric_logger.update(loss_d=loss_d_value)

        lr_d = optimizer_d.param_groups[0]["lr"]
        metric_logger.update(lr_d=lr_d)

        loss_d_value_reduce = misc.all_reduce_mean(loss_d_value)
        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_discriminator_loss', loss_d_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr_d', lr_d, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}