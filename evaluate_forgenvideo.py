import sys
sys.path.append('thirdparty/vidar/')
import os

import torch
from evaluation_utils.r3d3_argparser import argparser
from evaluation_utils.eval_dataset_forgenvideo import Evaluator


if __name__ == '__main__':
    os.environ['DIST_MODE'] = 'gpu' if torch.cuda.is_available() else 'cpu'
    args = argparser()

    wrapper = Evaluator(args)
    genvideo_root = "/gpfs/public-shared/fileset-groups/crosshair/guojiazhe/code/MagicDriveDiT-main/outputs/eval/CogVAE-848-17f/MagicDriveSTDiT3-XL-2_17-16x848x1600_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_map0_fsp8_cfg2.0_20241221-1102/generation/gen_video/b98e65c90a7a445a907a3bb1467b551b_gen0"
    first_sample_token = "b98e65c90a7a445a907a3bb1467b551b"
    wrapper.eval_scene_filepath(genvideo_root,6,first_sample_token,args.r3d3_image_size,[900,1600])
