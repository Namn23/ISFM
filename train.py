#!/usr/bin/python3
# coding=UTF-8
import argparse
import datetime
import logging
import os
import sys
import time
import warnings

import torch
import random
from accelerate import Accelerator
from accelerate.tracking import TensorBoardTracker, WandBTracker
from accelerate.utils import DistributedDataParallelKwargs, set_seed

from torch.utils.data import DataLoader

from datasets.MSRS_dataset_patch import MSRS_train, MSRS_test
from defaults import ISFM_C as cfg
from lr_scheduler import create_scheduler

from utils.utils import AverageMeter, postprocess_image, YCrCb2RGB
from loss import make_loss

from modeling.bulid_ISFM import build_model

warnings.filterwarnings("ignore")
logger = logging.getLogger()


def main(cfg):
    project_dir = os.path.join("./outputs", cfg.ACCELERATE.PROJECT_NAME)
    run_dir = os.path.join(project_dir, cfg.ACCELERATE.RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)

    accelerator = Accelerator(
        log_with=["wandb", "tensorboard"],
        project_dir=project_dir,
        mixed_precision=cfg.ACCELERATE.MIXED_PRECISION,
        gradient_accumulation_steps=cfg.ACCELERATE.GRADIENT_ACCUMULATION_STEPS,
        kwargs_handlers=[DistributedDataParallelKwargs(bucket_cap_mb=200, gradient_as_bucket_view=True)]
    )
    torch.backends.cuda.matmul.allow_tf32 = cfg.ACCELERATE.ALLOW_TF32
    set_seed(cfg.ACCELERATE.SEED)

    if accelerator.is_main_process:
        accelerator.trackers = []
        accelerator.trackers.append(WandBTracker(
            cfg.ACCELERATE.PROJECT_NAME, name=cfg.ACCELERATE.RUN_NAME, config=cfg, dir=project_dir))
        accelerator.trackers.append(TensorBoardTracker(cfg.ACCELERATE.RUN_NAME, project_dir))

        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            f.write(cfg.dump())
    accelerator.wait_for_everyone()

    fmt = "[%(asctime)s %(filename)s:%(lineno)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        filename=f"{run_dir}/log_rank{accelerator.process_index}.txt",
        filemode="a"
    )
    if accelerator.is_main_process:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(console_handler)

    logger.info(f"running with config:\n{str(cfg)}")

    logger.info("preparing datasets...")
    train_data = MSRS_train(
        root_dir=cfg.INPUT.ROOT_DIR,
        img_size=cfg.INPUT.IMG_SIZE,
        config_task=cfg.INPUT.TASK_CONFIG
    )

    logger.info(f"training with {train_data.num_vif} VIF images.")

    test_data = MSRS_test(
        root_dir=cfg.INPUT.ROOT_DIR,
        img_size=cfg.TEST.IMG_SIZE,
        config_task=cfg.INPUT.TASK_CONFIG
    )

    train_loader = DataLoader(
        train_data,
        cfg.INPUT.BATCH_SIZE // accelerator.num_processes,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.INPUT.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_data,
        cfg.TEST.MICRO_BATCH_SIZE,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=True
    )

    logger.info("preparing model...")
    weight_dtype = torch.float32
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    # elif accelerator.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16

    model = build_model(cfg)
    model.to(device=accelerator.device, dtype=weight_dtype)

    fusion_loss = make_loss().to(accelerator.device)
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info(f"number of trainable parameters: {trainable_params}")

    logger.info("preparing optimizer...")
    lr = cfg.OPTIMIZER.LR * cfg.INPUT.BATCH_SIZE if cfg.OPTIMIZER.SCALE_LR else cfg.OPTIMIZER.LR
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    #optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)

    logger.info("preparing accelerator...")
    model, optimizer, fusion_loss, train_loader, test_loader = accelerator.prepare(
        model, optimizer, fusion_loss, train_loader, test_loader)

    last_epoch = cfg.MODEL.LAST_EPOCH
    if cfg.MODEL.PRETRAINED_PATH:
        logger.info(f"loading states from {cfg.MODEL.PRETRAINED_PATH}")
        accelerator.load_state(cfg.MODEL.PRETRAINED_PATH)
    global_step = last_epoch * len(train_loader)

    logger.info("preparing lr_scheduler...")
    lr_scheduler = create_scheduler(cfg, optimizer, len(train_loader))

    logger.info("start training...")
    start_time = time.time()
    end_time = time.time()

    for epoch in range(last_epoch, cfg.OPTIMIZER.EPOCHS, 1):

        model.train()
        epoch_time = time.time()
        logger.info(f"epoch {epoch + 1} start")
        batch_time = AverageMeter()
        total_loss = AverageMeter()
        total_pixel = AverageMeter()
        total_grad = AverageMeter()
        total_max_pixel = AverageMeter()
        total_ssim = AverageMeter()

        if cfg.OPTIMIZER.SCHEDULER == "cosine":
            lr_scheduler.step(epoch)

        for i, batch in enumerate(train_loader):
            with ((accelerator.accumulate(model))):
                optimizer.zero_grad()

                jpg_names = batch["jpg_names"]
                with accelerator.autocast():
                    vi_y = batch["vi_y"]
                    ir = batch["ir"]
                    vi_src = batch["vi_src"]
                    ir_src = batch["ir_src"]
                    # src1 = F.interpolate(src1, size=(cfg.TEST.IMG_SIZE[0], cfg.TEST.IMG_SIZE[1]), mode='bilinear')
                    # src2 = F.interpolate(src2, size=(cfg.TEST.IMG_SIZE[0], cfg.TEST.IMG_SIZE[1]), mode='bilinear')
                    bsz = vi_y.shape[0]
                    outputs = model(batch)
                    fused_imgs = outputs
                    fused_imgs = fused_imgs.clamp(0, 1)
                    cr = batch["cr"]
                    cb = batch["cb"]
                    log_imgs = []
                    for k in range(fused_imgs.shape[0]):
                        log_img = YCrCb2RGB(fused_imgs[k], cr[k], cb[k])
                        log_imgs.append(log_img)
                    log_imgs = torch.stack(log_imgs, dim=0)
                LOSS_VIF = fusion_loss(vi_y, ir, fused_imgs, jpg_names, epoch)
                loss_fusion = LOSS_VIF["loss"]
                pixel_loss = LOSS_VIF["pixel_loss"]
                max_pixel_loss = LOSS_VIF["max_pixel_loss"]
                grad_loss = LOSS_VIF["grad_loss"]
                ssim_loss = LOSS_VIF["ssim_loss"]
                loss = loss_fusion

                # loss_fusion = F.l1_loss(fused_imgs, src1) + F.l1_loss(fused_imgs, src2)
                if torch.isnan(loss).any():
                    accelerator.set_trigger()
                if accelerator.check_trigger():
                    logger.info("loss is nan, stop training")
                    accelerator.end_training()
                    time.sleep(86400)  # waiting for...

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    global_step += 1
                optimizer.step()
                #lr_scheduler.step(epoch)
                if cfg.OPTIMIZER.SCHEDULER == "linear":
                    lr_scheduler.step()

            total_loss.update(loss_fusion.item())
            total_pixel.update(pixel_loss.item())
            total_max_pixel.update(max_pixel_loss.item())
            total_grad.update(grad_loss.item())
            total_ssim.update(ssim_loss.item())

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if (i + 1) % cfg.ACCELERATE.LOG_PERIOD == 0 or i == len(train_loader) - 1:
                accelerator.log({
                    "total_loss": total_loss.avg,
                    "pixel_loss": pixel_loss.item(),
                    "max_pixel_loss": max_pixel_loss.item(),
                    "grad_loss": grad_loss.item(),
                    "ssim_loss": total_ssim.avg,
                    "lr": optimizer.param_groups[-1]["lr"]
                }, step=global_step)

                if (epoch + 1) % 1 == 0: # cfg.ACCELERATE.EVAL_PERIOD
                    save_dir = os.path.join(run_dir, f"epochs_{(epoch + 1):03d}")
                    os.makedirs(save_dir, exist_ok=True)
                    log_dir = os.path.join(save_dir, "log_images")
                    os.makedirs(log_dir, exist_ok=True)

                    src1_imgs = vi_src
                    src2_imgs = ir_src
                    save_img = torch.stack([vi_src, ir_src, log_imgs])
                    save_img = postprocess_image(save_img, nrow=save_img.shape[0] * 2)
                    save_img.save(os.path.join(save_dir, f"training_test_{accelerator.process_index}_{i}.jpg"))

                    for j, jpg_name in zip(range(bsz), jpg_names):
                        generate_img = log_imgs[j:j + 1].detach()
                        # generate_img = F.interpolate(generate_img, size=(cfg.TEST.IMG_SIZE[0], cfg.TEST.IMG_SIZE[1])
                        #                              , mode='bilinear')
                        generate_img = generate_img.unsqueeze(0)
                        generate_img = postprocess_image(generate_img, nrow=1)
                        generate_img.save(os.path.join(log_dir, jpg_name))

                etas = batch_time.avg * (len(train_loader) - 1 - i)
                logger.info(
                    f"Train [{epoch + 1}/{cfg.OPTIMIZER.EPOCHS}]({i + 1}/{len(train_loader)})  "
                    f"Total Loss {loss.item():.4f}({total_loss.avg:.4f})  "
                    f"pixel_loss {pixel_loss.item():.4f}({total_pixel.avg:.4f})  "
                    f"max_pixel_loss {max_pixel_loss.item():.4f}({total_max_pixel.avg:.4f})  "
                    f"grad_loss {grad_loss.item():.4f}({total_grad.avg:.4f})  "
                    f"ssim_loss {ssim_loss.item():.4f}({total_ssim.avg:.4f})  "
                    f"Time {batch_time.val:.4f}({batch_time.avg:.4f})  "
                    f"Lr {optimizer.param_groups[-1]['lr']:.8f}  "
                    f"Eta {datetime.timedelta(seconds=int(etas))}")

        logger.info(
            f"epoch {epoch + 1} finished, total loss {total_loss.sum:.4f}({total_loss.avg:.4f}) , "
            f"pixel_loss {total_pixel.sum:.4f}({total_pixel.avg:.4f}),"
            f"grad_loss {total_grad.sum:.4f}({total_grad.avg:.4f}),"
            f"running time {datetime.timedelta(seconds=int(time.time() - epoch_time))}")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == 50 or (epoch + 1) == cfg.OPTIMIZER.EPOCHS:
            save_dir = os.path.join(run_dir, f"epochs_{(epoch + 1):03d}")
            save_dir = os.path.join(save_dir, "checkpoints")
            os.makedirs(save_dir, exist_ok=True)
            accelerator.save_state(save_dir)
            torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))

    train_time = time.time() - start_time
    logger.info(f'training completed, running time {datetime.timedelta(seconds=int(train_time))}')
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config_file", type=str, default="./configs/train.yaml", help="path to config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER
                        , help="modify config options using the command-line")
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    main(cfg)
