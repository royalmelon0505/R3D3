import os
import csv
import torch
from tqdm import tqdm
from lietorch import SE3

from r3d3.r3d3 import R3D3
from r3d3.modules.completion import DepthCompletion

from vidar.utils.write import write_npz
from vidar.utils.config import read_config, get_folder_name, load_class, Config, cfg_has
from vidar.utils.networks import load_checkpoint
from vidar.utils.config import recursive_assignment

from evaluation_utils.dataloader_wrapper import setup_dataloaders, SceneIterator, SampleIterator
from vidar.utils.setup import setup_metrics
from r3d3.utils import pose_matrix_to_quaternion

import cv2
import numpy as np
from evaluation_utils.pcd_vis import create_point_cloud_from_rgb_depth,colorize_depth_maps
from PIL import Image
import pickle
from pyquaternion import Quaternion

def get_tensor_from_video(video_path,dst_img_size):
    """
    :param video_path: 
    :return: pytorch tensor
    """
    if not os.access(video_path, os.F_OK):
        print('No exists')
        return

    import cv2

    cap = cv2.VideoCapture(video_path)

    frames_list = []
    while(cap.isOpened()):
        ret,frame = cap.read()

        if not ret:
            break
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, dsize=(dst_img_size[1], dst_img_size[0]), interpolation=cv2.INTER_LINEAR)
            frames_list.append(frame)
    cap.release()
    result_frames = torch.as_tensor(np.stack(frames_list))

    return result_frames

def load_completion_network(cfg: Config) -> DepthCompletion:
    """ Loads completion network with vidar framework
    Args:
        cfg: Completion network config
    Returns:
        Completion network with loaded checkpoint if path is provided
    """
    folder, name = get_folder_name(cfg.file, 'networks', root=cfg_has(cfg, 'root', 'vidar/arch'))
    network = load_class(name, folder)(cfg)
    recursive_assignment(network, cfg, 'networks', verbose=True)
    if cfg_has(cfg, 'checkpoint'):
        network = load_checkpoint(
            network,
            cfg.checkpoint,
            strict=False,
            verbose=True,
            prefix='completion'
        )
    return network.networks.cuda().eval()


class Evaluator:
    """ R3D3 evaluation module
    """
    def __init__(self, args):
        """
        Args:
            args: Arguments from argparser containing
                config: Path to vidar-config file (yaml) containing configurations for dataset, metrics (optional) and
                    completion network (optional)
                R3D3-args: As described by R3D3
                training_data_path: Path to directory where R3D3 training samples should be stored. If None, training
                    samples are not stored. Default - None
                prediction_data_path: Path to directory where R3D3 predictions should be stored. If None, predictions
                    are not stored. Default - None
        """
        self.args = args
        self.cfg = read_config(self.args.config)
        self.dataloaders = setup_dataloaders(self.cfg.datasets, n_workers=args.n_workers)
        self.completion_network = None
        if cfg_has(self.cfg, 'networks') and cfg_has(self.cfg.networks, 'completion'):
            self.completion_network = load_completion_network(
                self.cfg.networks.completion
            )
        self.metrics = {}
        if cfg_has(self.cfg, 'evaluation'):
            self.metrics = setup_metrics(self.cfg.evaluation)
        self.depth_results = []
        self.trajectory_results = []
        self.confidence_stats = []

        self.training_data_path = args.training_data_path
        self.prediction_data_path = args.prediction_data_path
        self.anno_path = "/gpfs/public-shared/fileset-groups/crosshair/guojiazhe/code/MagicDriveDiT-main/data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_track2_eval_with_bid.pkl"
        f=open(self.anno_path,'rb')
        self.nus_info=pickle.load(f)

    def eval_scene_filepath(self, genvideo_root: str, n_cams: int, first_sample_token:str,dst_res:list,ori_res:list) -> None:
        
        scene_depth_results = []
        pred_poses, gt_poses = [], []
        depth_res_idx = 0
        pred_pose_list = []
        pose_keys = ['x', 'y', 'z', 'r', 'i', 'j', 'k']

        d_ratio=[dst_res[0]/ori_res[0],dst_res[1]/ori_res[1]]
        r3d3 = R3D3(
            completion_net=self.completion_network,
            n_cams=n_cams,
            **{key.replace("r3d3_", ""): val for key, val in vars(self.args).items() if key.startswith("r3d3_")}
        )
        print("R3D3 init finish")
        # print(len(self.nus_info))
        video_dir = os.listdir(genvideo_root)
        cam_order_list = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
        
        first_sample_cnt=0
        for sample in self.nus_info['infos']:
            if first_sample_token == sample["token"]:
                print(first_sample_cnt)
                break
            first_sample_cnt+=1
        N_frame=16
        useful_clip = self.nus_info['infos'][first_sample_cnt:first_sample_cnt+N_frame]

        cam_6video_tensor=np.zeros((N_frame,6,dst_res[0],dst_res[1],3))
        for i in range(n_cams):
            cam_video_path = [item for item in video_dir if cam_order_list[i]+'.mp4' in item][0]
            # print(cam_video_path)
            cam_video_tensor = get_tensor_from_video(os.path.join(genvideo_root,cam_video_path),dst_res)
            cam_6video_tensor[:,i] = cam_video_tensor
            # print(cam_video_tensor.shape)
        cam_6video_tensor = torch.tensor(cam_6video_tensor) # T 6 H W 3
        cam_6video_tensor = cam_6video_tensor.permute(0,1,4,2,3)
        # print(cam_6video_tensor.shape)  #ideal T 6 3 H W

        cam_intrinsic_T_6=np.zeros((N_frame,6,4))
        cam_extrinsic_T_6=np.zeros((N_frame,6,4,4))
        i_t=0
        for frame_info in useful_clip:
            # print(frame_info['cams'].keys())
            i_c=0
            cam_info_6=frame_info['cams']
            for cam_type in cam_order_list:
                cam_info = cam_info_6[cam_type]
                cam_intrinsic = cam_info['camera_intrinsics']
                cam_intrinsic_T_6[i_t,i_c,:]=np.array([cam_intrinsic[0,0]*d_ratio[0],cam_intrinsic[1,1]*d_ratio[1],cam_intrinsic[0,2]*d_ratio[0],cam_intrinsic[1,2]*d_ratio[1]])
                
                cam2ego_r = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
                cam2ego_t = np.array(cam_info['sensor2ego_translation']).T
                cam2ego_rt = np.eye(4)
                cam2ego_rt[:3, :3] = cam2ego_r
                cam2ego_rt[3, :3] = cam2ego_t
                # cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
                # print(cam2ego_rt)
                cam_extrinsic_T_6[i_t,i_c,:,:]=cam2ego_rt
                i_c+=1
            i_t+=1

        cam_extrinsic_T_6 = torch.tensor(cam_extrinsic_T_6)
        cam_intrinsic_T_6 = torch.tensor(cam_intrinsic_T_6)
        # print(cam_intrinsic_T_6[0])
        # intrinsics_T_6 = np.array( cam_intrinsic_T_6[0, :, [0, 1, 0, 1], [0, 1, 2, 2]] )
        # print(intrinsics_T_6.shape)
        # import pdb ; pdb.set_trace()

        for timestamp in range(N_frame):
            cam_e_ti = cam_extrinsic_T_6[timestamp]
            cam_i_ti = cam_intrinsic_T_6[timestamp]
            rgb_ti = cam_6video_tensor[timestamp]
            # print(cam_e_ti.shape,cam_i_ti.shape,rgb_ti.shape)
            pose = SE3(pose_matrix_to_quaternion(cam_e_ti).cuda())
            pose = pose.inv()
            pose_rel = (pose * pose[0:1].inv())

            output = r3d3.track(
                tstamp=float(timestamp),
                image=(rgb_ti).type(torch.uint8).cuda(),
                intrinsics=cam_i_ti.cuda(),
                mask=None,
                pose_rel=pose_rel.data
            )
            # print(output.shape)

                                            
        # for timestamp, sample in enumerate(tqdm(sample_iterator, desc='Sample', position=0, leave=True)):
        #     pose = SE3(pose_matrix_to_quaternion(sample['pose'][0][0]).cuda())
        #     pose = pose.inv()
        #     pose_rel = (pose * pose[0:1].inv())

        #     intrinsics = sample['intrinsics'][0][0, :, [0, 1, 0, 1], [0, 1, 2, 2]]
        #     is_keyframe = 'depth' in sample and sample['depth'][0].max() > 0.

        #     # import pdb ; pdb.set_trace()
        #     # print(sample['depth'][0].shape)
        #     output = r3d3.track(
        #         tstamp=float(timestamp),
        #         image=(sample['rgb'][0][0] * 255).type(torch.uint8).cuda(),
        #         intrinsics=intrinsics.cuda(),
        #         mask=(sample['mask'][0][0, :, 0] > 0).cuda() if 'mask' in sample else None,
        #         pose_rel=pose_rel.data
        #     )

            output = {key: data.cpu() if torch.is_tensor(data) else data for key, data in output.items()}
            pred_pose = None


            # if output['pose'] is not None:
            #     pred_pose = (pose_rel.cpu() * SE3(output['pose'][None])).inv()
            #     pred_pose_list.append(
            #         {'filename': sample['filename'][0][0][0], **dict(zip(pose_keys, pred_pose[0].data.numpy()))}
            #     )
            #     pred_poses.append(pred_pose.matrix())
            #     gt_poses.append(sample['pose'][0][0])
            # if output['disp_up'] is not None and 'depth' in self.metrics and is_keyframe:
            #     results = {
            #         'ds_idx': sample['idx'][0],
            #         'sc_idx': torch.tensor(depth_res_idx, dtype=sample['idx'][0].dtype, device=sample['idx'][0].device),
            #         'scene': scene
            #     }
            #     results.update({key: metric[0] for key, metric in self.metrics['depth'].evaluate(
            #         batch=sample,
            #         output={'depth': {0: [1 / output['disp_up'].unsqueeze(0).unsqueeze(2)]}}
            #     )[0].items()})
            #     scene_depth_results.append(results)
            #     depth_res_idx += 1

            # if self.training_data_path is not None and pred_pose is not None:
            #     for cam, filename in enumerate(sample['filename'][0]):
            #         write_npz(
            #             os.path.join(
            #                 self.training_data_path,
            #                 filename[0].replace('rgb', 'r3d3').replace('CAM_', 'R3D3_') + '.npz'
            #             ),
            #             {
            #                 'intrinsics': intrinsics[cam].numpy(),
            #                 'pose': pred_pose[cam].data.numpy(),
            #                 'disp': output['disp'][cam].numpy()[None],
            #                 'disp_up': output['disp_up'][cam].numpy()[None],
            #                 'conf': output['conf'][cam].numpy()[None],
            #             }
            #         )
            if self.prediction_data_path is not None and output['disp_up'] is not None :
                cam_i=0
                # for cam, filename in enumerate(sample['filename'][0]):
                for cam in range(6):
                    filename=[f"{cam_order_list[cam]}/{timestamp}"]
                    # write_npz(
                    #     os.path.join(
                    #         self.prediction_data_path,
                    #         filename[0] + '_depth(0)_pred.npz'
                    #     ),
                    #     {
                    #         'depth': (1.0 / output['disp_up'][cam].numpy()),
                    #         'intrinsics': intrinsics[cam].numpy(),
                    #         'd_info': 'r3d3_depth',
                    #         't': float(timestamp)
                    #     }
                    # )
                    # print(output['disp_up'].shape)
                    pred_depth = (1.0 / output['disp_up'][cam].numpy())
                    # rgb = np.transpose(sample['rgb'][0][0][cam].numpy(),(1,2,0)) 
                    rgb = np.transpose(rgb_ti[cam].numpy(),(1,2,0)) 
                    r,g,b = cv2.split(rgb)
                    bgr = cv2.merge([b,g,r])

                    vis_root = "/gpfs/public-shared/fileset-groups/crosshair/guojiazhe/code/r3d3/data/pred_data_path/vis_gen_video"
                    os.makedirs(os.path.join(vis_root,filename[0]),exist_ok=True)
                    cv2.imwrite(os.path.join(vis_root,filename[0],'rgb.png'),bgr)

                    im_color = colorize_depth_maps(pred_depth,0.1,80)
                    im=Image.fromarray(np.uint8(im_color[0]*255))
                    im.save(os.path.join(vis_root,filename[0],'depth.png'))
                    
                    # s_depth = sample["depth"][0][0][cam].numpy()
                    # np.save(os.path.join(vis_root,filename[0],'depth_sparse.npy'),s_depth)

                    # sv_depth=cv2.split(s_depth)[0]
                    # sv_depth[sv_depth>80]=0
                    # sv_depth=sv_depth/80
                    # print(s_depth.shape)
                    # cv2.imwrite(os.path.join(vis_root,filename[0],'depth_sparse.png'),sv_depth[0]*255)

                    # print(intrinsics[cam].numpy())

                    # create_point_cloud_from_rgb_depth(rgb*255,pred_depth,intrinsics[cam].numpy().tolist(),os.path.join(vis_root,filename[0],'out_sv.pcd'))
                    
                    # print(sample['rgb'][0].shape,pred_depth.shape)
                    # cam_i+=1

        # Terminate
        del r3d3
        torch.cuda.empty_cache()

        # if self.prediction_data_path is not None and len(pred_pose_list) > 0:
        #     pose_dir = os.path.join(self.prediction_data_path, 'poses')
        #     if not os.path.exists(pose_dir):
        #         os.makedirs(pose_dir)
        #     with open(os.path.join(pose_dir, f'{scene}_poses.csv'), 'w') as csvfile:
        #         writer = csv.DictWriter(csvfile, fieldnames=pred_pose_list[0].keys())
        #         writer.writeheader()
        #         writer.writerows(pred_pose_list)
        # self.depth_results.extend([{'idx': res['ds_idx'], **res} for res in scene_depth_results])
        # if 'depth' in self.metrics and len(scene_depth_results) >= 1:
        #     reduced_data = self.metrics['depth'].reduce_metrics(
        #         [[{'idx': res['sc_idx'], **res} for res in scene_depth_results]],
        #         [scene_depth_results], strict=False
        #     )
        #     self.metrics['depth'].print(reduced_data, [f'scene-{scene}'])
        # if 'trajectory' in self.metrics and len(gt_poses) >= 2:
        #     results = {'scene': scene}
        #     results.update(self.metrics['trajectory'].evaluate(
        #         batch={'trajectory': {0: torch.stack(gt_poses)}},
        #         output={'trajectory': {0: [torch.stack(pred_poses)]}}
        #     )[0])
        #     self.trajectory_results.append({'idx': torch.tensor(len(self.trajectory_results)), **results})
        #     reduced_data = self.metrics['trajectory'].reduce_metrics(
        #         [[{'idx': torch.tensor(0), **results}]],
        #         [[results]], strict=False
        #     )
        #     self.metrics['trajectory'].print(reduced_data, [f'scene-{scene}'])



    # def eval_datasets(self) -> None:
    #     """ Evaluates datasets consisting of multiple scenes
    #     """
    #     for dataloader in tqdm(self.dataloaders, desc='Datasets', position=2, leave=True):
    #         n_cams = len(dataloader.dataset.cameras)
    #         pbar = tqdm(SceneIterator(dataloader), desc='Scenes', position=1, leave=True)
    #         for scene, sample_iterator in pbar:
    #             pbar.set_postfix_str("Processing Scene - {}".format(scene))
    #             pbar.refresh()
    #             self.eval_scene(scene, n_cams, sample_iterator)

    #         if 'depth' in self.metrics and len(self.depth_results) > 0:
    #             reduced_data = self.metrics['depth'].reduce_metrics(
    #                 [self.depth_results],
    #                 [dataloader.dataset], strict=False
    #             )
    #             self.metrics['depth'].print(reduced_data, ['Overall'])
    #         if 'trajectory' in self.metrics and len(self.trajectory_results) > 0:
    #             reduced_data = self.metrics['trajectory'].reduce_metrics(
    #                 [self.trajectory_results],
    #                 [self.trajectory_results], strict=False
    #             )
    #             self.metrics['trajectory'].print(reduced_data, ['Overall'])
    #         # Use to evaluate confidence statistics => Can find scenes where metric was not recovered / failed
    #         # if len(self.confidence_stats) > 0:
    #         #     confidence_stats_summary = {}
    #         #     for element in self.confidence_stats:
    #         #         scene = element['scene']
    #         #         if scene not in confidence_stats_summary:
    #         #             stats_keys = [key for key in element.keys() if key is not scene]
    #         #             confidence_stats_summary[scene] = {k: [] for k in stats_keys if k not in ['scene', 'idx']}
    #         #         for key in confidence_stats_summary[scene]:
    #         #             confidence_stats_summary[scene][key].append(element[key])
    #         #     confidence_stats_summary = {
    #         #         scene: {key: sum(val) / len(val) for key, val in stats.items()}
    #         #         for scene, stats in confidence_stats_summary.items()
    #         #     }
    #         #     import csv
    #         #     with open('confidence_stats.csv', 'w', newline='') as output_file:
    #         #         dict_writer = csv.DictWriter(output_file, list(self.confidence_stats[0].keys()))
    #         #         dict_writer.writeheader()
    #         #         dict_writer.writerows(self.confidence_stats)
