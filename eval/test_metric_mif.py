import os
from .Evaluator import Evaluator
import numpy as np
import cv2
import time
import datetime
from utils import AverageMeter


def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img


# test_folder = './data/MSRS/test'
# test_out_folder = './test_result_159_640_480'
def evaluate(logger, length, test_folder='./data/MSRS/test_512_512', test_fuse_folder='./test_result_159', task="PET-MRI"):
    metric_result = np.zeros((9))
    end_time = time.time()
    batch_time = AverageMeter()

    if task == "PET-MRI":
        # src1_path = test_folder + "/PET-MRI/test/PET"
        # src2_path = test_folder + "/PET-MRI/test/MRI"
        src1_path = test_folder + "/PET-MRI/test/vi"
        src2_path = test_folder + "/PET-MRI/test/ir"
    elif task == "CT-MRI":
        # src1_path = test_folder + "/CT-MRI/test/CT"
        # src2_path = test_folder + "/CT-MRI/test/MRI"
        src1_path = test_folder + "/CT-MRI/test/vi"
        src2_path = test_folder + "/CT-MRI/test/ir"
    else:
        assert task == "SPECT-MRI"
        # src1_path = test_folder + "/SPECT-MRI/test/SPECT"
        # src2_path = test_folder + "/SPECT-MRI/test/MRI"
        src1_path = test_folder + "/SPECT-MRI/test/vi"
        src2_path = test_folder + "/SPECT-MRI/test/ir"

    i = 0
    path1 = sorted(os.listdir(os.path.join(src1_path)))
    path2 = sorted(os.listdir(os.path.join(src2_path)))
    for img_name1, img_name2 in zip(path1, path2):
        img_name = img_name1.split('_')[0]
        img_name = img_name + '.jpg' #
        fused_name = task + "_" + img_name1

        vi = image_read_cv2(os.path.join(src1_path, img_name1), 'GRAY')
        ir = image_read_cv2(os.path.join(src2_path, img_name2), 'GRAY')
        fi = image_read_cv2(os.path.join(test_fuse_folder, fused_name), 'GRAY')

        metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                      , Evaluator.SF(fi), Evaluator.AG(fi), Evaluator.MI(fi, ir, vi)
                                      , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                      , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)])
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (i + 1) % 20 == 0 or (i + 1) == length:
            etas = batch_time.avg * (length - 1 - i)
            logger.info(
                f"Testing Metrics ({i + 1}/{length})  "
                f"Time {batch_time.val:.4f}({batch_time.avg:.4f})  "
                f"Eta {datetime.timedelta(seconds=int(etas))}")

        i += 1
    logger.info(f"Evaluation {task} Metrics Results")
    metric_result /= len(os.listdir(test_fuse_folder))
    logger.info("\t\t EN\t SD\t SF\t AG\t MI\tSCD\tVIF\tQabf\tSSIM")
    logger.info('MambaFusion' + '\t' + str(np.round(metric_result[0], 2)) + '\t'
                + str(np.round(metric_result[1], 2)) + '\t'
                + str(np.round(metric_result[2], 2)) + '\t'
                + str(np.round(metric_result[3], 2)) + '\t'
                + str(np.round(metric_result[4], 2)) + '\t'
                + str(np.round(metric_result[5], 2)) + '\t'
                + str(np.round(metric_result[6], 2)) + '\t'
                + str(np.round(metric_result[7], 2)) + '\t'
                + str(np.round(metric_result[8], 2))
                )
    logger.info("=" * 80)

    EN = np.round(metric_result[0], 2)
    SD = np.round(metric_result[1], 2)
    SF = np.round(metric_result[2], 2)
    AG = np.round(metric_result[3], 2)
    MI = np.round(metric_result[4], 2)
    SCD = np.round(metric_result[5], 2)
    VIF = np.round(metric_result[6], 2)
    Qabf = np.round(metric_result[7], 2)
    SSIM = np.round(metric_result[8], 2)
    Metrics = [EN, SD, SF, AG, MI, SCD, VIF, Qabf, SSIM]

    return Metrics

