#!/usr/bin/python3
# coding=UTF-8
import argparse
import copy
import datetime
import logging
import os
import random
import sys
import time
import warnings

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.tracking import TensorBoardTracker, WandBTracker
from accelerate.utils import DistributedDataParallelKwargs, set_seed

from torch.utils.data import DataLoader

from datasets.MSRS_dataset_patch import (MSRS_test, MSRS_train)
from defaults import ISFM_C as cfg

from utils import AverageMeter, postprocess_image
from utils.utils import YCrCb2RGB
from modeling.bulid_ISFM import build_model
from eval.test_metric import evaluate

warnings.filterwarnings("ignore")
logger = logging.getLogger()


def do_test(cfg, model, test_loader, save_dir, global_step, accelerator, task_type):

    model.eval()
    if 'VIF' in task_type:
        root_dir = cfg.INPUT.ROOT_DIR
    elif 'MIF' in task_type:
        root_dir = cfg.TEST.MIF_DIR
    elif 'MEF' in task_type:
        root_dir = cfg.TEST.MEF_DIR
    elif 'MFF' in task_type:
        root_dir = cfg.TEST.MFF_DIR
    else:
        raise ValueError('wrong test type!')
    logger.info(f"Testing from path {root_dir} ")
    with torch.no_grad():
        end_time = time.time()
        batch_time = AverageMeter()

        for i, test_batch in enumerate(test_loader):

            jpg_names = test_batch["jpg_names"]
            with accelerator.autocast():
                vi_y = test_batch["vi_y"]
                ir = test_batch["ir"]
                vi_src = test_batch["vi_src"]
                ir_src = test_batch["ir_src"]
                bsz = vi_y.shape[0]
                outputs = model(test_batch)
                fused_imgs = outputs
                fused_imgs = fused_imgs.clamp(0, 1)
                cr = test_batch["cr"]
                cb = test_batch["cb"]
                log_imgs = []
                for k in range(fused_imgs.shape[0]):
                    log_img = YCrCb2RGB(fused_imgs[k], cr[k], cb[k])
                    log_imgs.append(log_img)
                fused_imgs = torch.stack(log_imgs, dim=0)

            log_img = torch.stack([vi_src, ir_src, fused_imgs])
            log_img = postprocess_image(log_img, nrow=log_img.shape[0] * 2)
            log_img.save(os.path.join(save_dir, f"log_test_{accelerator.process_index}_{i}.jpg"))

            test_dir = os.path.join(save_dir, "test_results")
            os.makedirs(test_dir, exist_ok=True)
            for j, jpg_name in zip(range(bsz), jpg_names):
                save_img = fused_imgs[j:j + 1]

                save_img = save_img.unsqueeze(0)
                save_img = postprocess_image(save_img, nrow=1)
                save_img.save(os.path.join(test_dir, jpg_name))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if (i + 1) % cfg.ACCELERATE.LOG_PERIOD == 0 or i == len(test_loader) - 1:
                etas = batch_time.avg * (len(test_loader) - 1 - i)
                logger.info(
                    f"Testing Results({i + 1}/{len(test_loader)})  "
                    f"Time {batch_time.val:.4f}({batch_time.avg:.4f})  "
                    f"Eta {datetime.timedelta(seconds=int(etas))}")
                if os.environ.get("WANDB_MODE", None) == "offline":
                    break

        length = len(test_loader) * cfg.TEST.MICRO_BATCH_SIZE
        if accelerator.is_main_process:
            Test_VIF_Metrics = evaluate(logger, length, root_dir, test_dir, task=task_type)

            logger.info("Evaluation Results:")
            logger.info(f"EN: {Test_VIF_Metrics[0]:.2f}")
            logger.info(f"SD: {Test_VIF_Metrics[1]:.2f}")
            logger.info(f"SF: {Test_VIF_Metrics[2]:.2f}")
            logger.info(f"AG: {Test_VIF_Metrics[3]:.2f}")
            logger.info(f"MI: {Test_VIF_Metrics[4]:.2f}")
            logger.info(f"SCD: {Test_VIF_Metrics[5]:.2f}")
            logger.info(f"VIF: {Test_VIF_Metrics[6]:.2f}")
            logger.info(f"Qabf: {Test_VIF_Metrics[7]:.2f}")
            logger.info(f"SSIM: {Test_VIF_Metrics[8]:.2f}")

            accelerator.log({
                "EN": Test_VIF_Metrics[0],
                "SD": Test_VIF_Metrics[1],
                "SF": Test_VIF_Metrics[2],
                "AG": Test_VIF_Metrics[3],
                "MI": Test_VIF_Metrics[4],
                "SCD": Test_VIF_Metrics[5],
                "VIF": Test_VIF_Metrics[6],
                "Qabf": Test_VIF_Metrics[7],
                "SSIM": Test_VIF_Metrics[8]
            }, step=global_step)

        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()

def test(cfg):
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
        filename=f"{run_dir}/test_log_rank{accelerator.process_index}.txt",
        filemode="a"
    )
    if accelerator.is_main_process:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(console_handler)

    logger.info(f"running with config:\n{str(cfg)}")

    logger.info("preparing datasets...")

    test_data = MSRS_test(
        root_dir=cfg.INPUT.ROOT_DIR,
        img_size=cfg.TEST.IMG_SIZE,
        config_task=cfg.TEST.TASK_CONFIG
    )

    test_loader = DataLoader(
        test_data,
        cfg.TEST.MICRO_BATCH_SIZE,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=True
    )

    logger.info("preparing model...")
    weight_dtype = torch.float32

    model = build_model(cfg)
    model.to(device=accelerator.device, dtype=weight_dtype)

    chec_dir = cfg.TEST.CHECKPOINT_PATH

    logger.info(f"loading checkpoints from {chec_dir}")
    logger.info(model.load_state_dict(torch.load(
        os.path.join(chec_dir, "model.pth"), map_location="cpu"
    ), strict=False))

    logger.info("preparing accelerator...")
    model, test_loader = accelerator.prepare(
        model, test_loader)

    logger.info(f"start testing {cfg.TEST.TASK_CONFIG[0]}...")
    save_dir = os.path.join(run_dir, "log_images")
    save_dir = os.path.join(save_dir, 'best')
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"saving to {save_dir}")

    do_test(
        cfg=cfg,
        model=model,
        test_loader=test_loader,
        save_dir=save_dir,
        global_step=None,
        accelerator=accelerator,
        task_type=cfg.TEST.TASK_CONFIG[0]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument("--config_file", type=str, default="./configs/test.yaml", help="path to config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER
                        , help="modify config options using the command-line")
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    test(cfg)
