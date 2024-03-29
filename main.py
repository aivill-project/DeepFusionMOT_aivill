# -*-coding:utf-8-*
# author: wangxy
from __future__ import print_function
import os, numpy as np, time, cv2, torch
from os import listdir
from os.path import join
from file_operation.file import load_list_from_folder, mkdir_if_inexistence, fileparts
from detection.detection import Detection_2D, Detection_3D_only, Detection_3D_Fusion
from tracking.tracker import Tracker
from datasets.datafusion import datafusion2Dand3D
from datasets.coordinate_transformation import convert_3dbox_to_8corner, convert_x1y1x2y2_to_tlwh
from visualization.visualization_3d import show_image_with_boxes
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='datasets/kitti/train', help='dataset path')
parser.add_argument('--detections3D', type=str, default='3D_pointrcnn_Car_val', help='detections 3D foldername')
parser.add_argument('--detections2D', type=str, default='2D_rrc_Car_val', help='detections 2D foldername')
parser.add_argument('--save-path', type=str, default='results/train', help='result save path')
parser.add_argument('--save-img', action='store_true', help='save image')
parser.add_argument('--eval', action='store_true', help='start evaluation')
parser.add_argument('--gt-data', type=str, default=None, help='gt data dataframe path')
# parser.add_argument('--eval-output', type=str, default=None, help='eval result save path')
args = parser.parse_args()

# data_root = 'datasets/kitti/train'
# detections_name_3D = '3D_pointrcnn_Car_val'
# detections_name_2D = '2D_rrc_Car_val'



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


class DeepFusion(object):
    def __init__(self, max_age, min_hits):
        '''
        :param max_age:  The maximum frames in which an object disappears.
        :param min_hits: The minimum frames in which an object becomes a trajectory in succession.
        '''
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracker = Tracker(max_age,min_hits)
        self.reorder = [3, 4, 5, 6, 2, 1, 0]
        self.reorder_back = [6, 5, 4, 0, 1, 2, 3]
        self.frame_count = 0

    def update(self,detection_3D_fusion,detection_2D_only,detection_3D_only,detection_3Dto2D_only,
               additional_info, calib_file):

        dets_3d_fusion = np.array(detection_3D_fusion['dets_3d_fusion'])
        dets_3d_fusion_info = np.array(detection_3D_fusion['dets_3d_fusion_info'])
        dets_3d_only = np.array(detection_3D_only['dets_3d_only'])
        dets_3d_only_info = np.array(detection_3D_only['dets_3d_only_info'])

        if len(dets_3d_fusion) == 0:
            dets_3d_fusion = dets_3d_fusion
        else:
            dets_3d_fusion = dets_3d_fusion[:,self.reorder]  # convert [h,w,l,x,y,z,rot_y] to [x,y,z,rot_y，l,w,h]
        if len(dets_3d_only) == 0:
            dets_3d_only = dets_3d_only
        else:
            dets_3d_only = dets_3d_only[:, self.reorder]

        detection_3D_fusion = [Detection_3D_Fusion(det_fusion, dets_3d_fusion_info[i]) for i, det_fusion in enumerate(dets_3d_fusion)]
        detection_3D_only = [Detection_3D_only(det_only, dets_3d_only_info[i]) for i, det_only in enumerate(dets_3d_only)]
        detection_2D_only = [Detection_2D(det_fusion) for i, det_fusion in enumerate(detection_2D_only)]

        self.tracker.predict_2d()
        self.tracker.predict_3d()
        self.tracker.update(detection_3D_fusion, detection_3D_only, detection_3Dto2D_only, detection_2D_only, calib_file, iou_threshold=0.5)
        # self.tracker.update(detection_3D_fusion, detection_3D_only, detection_3Dto2D_only, detection_2D_only, calib_file, iou_threshold=0.01)

        self.frame_count += 1
        outputs = []
        for track in self.tracker.tracks_3d:
            if track.is_confirmed():
                bbox = np.array(track.pose[self.reorder_back])
                outputs.append(np.concatenate(([track.track_id_3d], bbox, track.additional_info)).reshape(1, -1))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):  # Convert the coordinate format of the bbox box from center x, y, w, h to upper left x, upper left y, w, h
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _tlwh_to_xyxy(self, bbox_tlwh):
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), 0)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), 0)
        return x1, y1, x2, y2

    def _tlwh_to_x1y1x2y2(self, bbox_tlwh):
        x, y, w, h = bbox_tlwh
        x1 = x
        x2 = x + w
        y1 = y
        y2 = y + h
        return x1, y1, x2, y2


if __name__ == '__main__':
    # Define the file name
    # data_root = 'datasets/kitti/train'
    # detections_name_3D = '3D_pointrcnn_Car_val'
    # detections_name_2D = '2D_rrc_Car_val'
    data_root = args.data_path
    detections_name_3D = args.detections3D
    detections_name_2D = args.detections2D

    # Define the file path
    # calib_root = os.path.join(data_root, 'calib_train')
    # dataset_dir = os.path.join(data_root,'image_02_train')
    calib_root = os.path.join(data_root, 'calib')
    dataset_dir = os.path.join(data_root,'image_02')
    detections_root_3D = os.path.join(data_root, detections_name_3D)
    detections_root_2D = os.path.join(data_root, detections_name_2D)

    # Define the file path of results.
    # save_root = 'results/train'   # The root directory where the result is saved
    save_root = args.save_path
    txt_path_0 = os.path.join(save_root, 'data'); mkdir_if_inexistence(txt_path_0)
    image_path_0 = os.path.join(save_root, 'image'); mkdir_if_inexistence(image_path_0)
    # Open file to save in list.
    # det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
    # det_id2str = {1: 'Car', 2: 'Two_Wheeler', 3: 'Adult', 4: 'Kid', 5: 'SUV', 6: 'Van'}
    det_id2str = {1: 'Car', 2: 'SUV_&_Van', 3: 'Truck', 4: 'Bus', 5: 'Special_Vehicle', 6: 'Two_Wheeler', 7: 'Person'}
    # det_id2str = {1: 'Small_Car',
    #                 2: 'Light_Car',
    #                 3: 'Car',
    #                 4: 'Van',
    #                 5: 'SUV',
    #                 6: 'Small_Truck',
    #                 7: 'Medium_Truck',
    #                 8: 'Large_Truck',
    #                 9: 'Mini_Bus',
    #                 10: 'Bus',
    #                 11: 'Special_Vehicle',
    #                 12: 'Two_Wheeler',
    #                 13: 'Kickboard',
    #                 14: 'Adult',
    #                 15: 'Kid'}

    calib_files = sorted(os.listdir(calib_root))
    detections_files_3D = sorted(os.listdir(detections_root_3D))
    detections_files_2D = sorted(os.listdir(detections_root_2D))
    image_files = sorted(os.listdir(dataset_dir))
    detection_file_list_3D, num_seq_3D = load_list_from_folder(detections_files_3D, detections_root_3D)
    # print(detection_file_list_3D)
    detection_file_list_2D, num_seq_2D = load_list_from_folder(detections_files_2D, detections_root_2D)
    image_file_list, _ = load_list_from_folder(image_files, dataset_dir)

    total_time, total_frames, i = 0.0, 0, 0  # Tracker runtime, total frames and Serial number of the dataset
    tracker = DeepFusion(max_age=40, min_hits=1)  # Tracker initialization

    # Iterate through each data set
    for seq_file_3D, image_filename in zip(detection_file_list_3D, image_files):
        print('--------------Start processing the {} dataset--------------'.format(image_filename))
        total_image = 0  # Record the total frames in this dataset
        seq_file_2D = detection_file_list_2D[i]
        seq_name, datasets_name, _ = fileparts(seq_file_3D)
        txt_path = txt_path_0 + "/" + image_filename + '.txt'
        image_path = image_path_0 + '/' + image_filename; mkdir_if_inexistence(image_path)

        calib_file = [calib_file for calib_file in calib_files if calib_file==seq_name ]
        calib_file_seq = os.path.join(calib_root, ''.join(calib_file))
        image_dir = os.path.join(dataset_dir, image_filename)
        image_filenames = sorted([join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)])
        seq_dets_3D = np.loadtxt(seq_file_3D, delimiter=',').reshape(-1, 15)  # load 3D detections, N x 15
        seq_dets_2D = np.loadtxt(seq_file_2D, delimiter=',').reshape(-1, 7)  # load 2D detections, N x 6

        # min_frame, max_frame = int(seq_dets_3D[:, 0].min()), len(image_filenames)
        select_frames = list(set(seq_dets_3D[:, 0].astype(int).flatten().tolist()))
        image_filenames = np.asarray(image_filenames)[select_frames]
        # print(select_frames)
        # print(np.asarray(image_filenames))
        # print(image_filenames)

        # for frame, img0_path in zip(range(min_frame, max_frame + 1), image_filenames):
        for frame, img0_path in zip(select_frames, image_filenames):
            # print(img0_path)
            img_0 = cv2.imread(img0_path)
            _, img0_name, _ = fileparts(img0_path)
            dets_3D_camera = seq_dets_3D[seq_dets_3D[:, 0] == frame, 7:14]  # 3D bounding box(h,w,l,x,y,z,theta)
            dets_8corners = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets_3D_camera]

            ori_array = seq_dets_3D[seq_dets_3D[:, 0] == frame, -1].reshape((-1, 1))
            other_array = seq_dets_3D[seq_dets_3D[:, 0] == frame, 1:7]
            additional_info = np.concatenate((ori_array, other_array), axis=1)

            dets_3Dto2D_image = seq_dets_3D[seq_dets_3D[:, 0] == frame, 2:6]
            # print(seq_dets_2D.shape)
            # dets_2D = seq_dets_2D[seq_dets_2D[:, 0] == frame, 1:5]   # 2D bounding box(x1,y1,x2,y2)
            dets_2D = seq_dets_2D[seq_dets_2D[:, 0] == frame, 1:6]   # 2D bounding box(class, x1,y1,x2,y2)

            # Data Fusion(3D and 2D detections)
            # detection_2D_fusion, detection_3Dto2D_fusion, detection_3D_fusion, detection_2D_only, detection_3Dto2D_only, detection_3D_only = \
            #     datafusion2Dand3D(dets_3D_camera, dets_2D, dets_3Dto2D_image, additional_info)
            detection_2D_fusion, detection_3Dto2D_fusion, detection_3D_fusion, detection_2D_only, detection_3Dto2D_only, detection_3D_only, \
            new_additional_info \
                = datafusion2Dand3D(dets_3D_camera, dets_2D, dets_3Dto2D_image, additional_info)
            detection_2D_only_tlwh = np.array([convert_x1y1x2y2_to_tlwh(i) for i in detection_2D_only]) # (x1,y1,x2,y2) to (x,y,center_x,center_y)

            # print('detection_2D_fusion:', len(detection_2D_fusion))
            # print('detection_3Dto2D_fusion:', len(detection_3Dto2D_fusion))
            # print('detection_3D_fusion:', len(detection_3D_fusion['dets_3d_fusion']))
            # print('detection_3Dto2D_only:', detection_3Dto2D_only.shape)
            # print('detection_3D_only:', detection_3D_only['dets_3d_only'])
            # print('detection_2D_only:', detection_2D_only.shape)
            # print('-' * 50)
            
             # just using fusion data
            detection_3Dto2D_only = []
            detection_3D_only = {'dets_3d_only': [], 'dets_3d_only_info': []}

            start_time = time.time()
            trackers = tracker.update(detection_3D_fusion, detection_2D_only_tlwh, detection_3D_only, detection_3Dto2D_only,
                                      additional_info, calib_file_seq)
            # trackers = tracker.update(detection_3D_fusion, detection_2D_only, detection_3D_only, detection_3Dto2D_only,
            #                           new_additional_info, calib_file_seq)
            cycle_time = time.time() - start_time
            total_time += cycle_time

            # Outputs
            total_frames += 1 # Total frames for all datasets
            total_image += 1 # Total frames for a dataset
            if total_image % 50 == 0:
                print("Now start processing the {} image of the {} dataset".format(total_image, image_filename))

            if len(trackers) > 0:
                for d in trackers:
                    bbox3d = d.flatten()
                    bbox3d_tmp = bbox3d[1:8]  # 3D bounding box(h,w,l,x,y,z,theta)
                    id_tmp = int(bbox3d[0])
                    # id_tmp = int(bbox3d[0]) - int(trackers[0].flatten()[0]) # 라벨 번호 0부터 시작하기
                    ori_tmp = bbox3d[8]
                    type_tmp = det_id2str[bbox3d[9]]
                    bbox2d_tmp_trk = bbox3d[10:14]
                    conf_tmp = bbox3d[14]
                    color = compute_color_for_id(id_tmp)
                    # label = f'{id_tmp} {"car"}'
                    label = f'{id_tmp} {type_tmp}'
                    image_save_path = os.path.join(image_path, '%04d.jpg' % (int(img0_name)))
                    with open(txt_path, 'a') as f:
                        str_to_srite = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (frame, id_tmp,type_tmp, ori_tmp,bbox2d_tmp_trk[0],
                                bbox2d_tmp_trk[1],bbox2d_tmp_trk[2],bbox2d_tmp_trk[3],bbox3d_tmp[0], bbox3d_tmp[1],bbox3d_tmp[2], bbox3d_tmp[3],
                                bbox3d_tmp[4], bbox3d_tmp[5],bbox3d_tmp[6],conf_tmp)
                        f.write(str_to_srite)
                        if args.save_img:
                            show_image_with_boxes(img_0, bbox3d_tmp, image_path, color, img0_name, label, calib_file_seq,line_thickness=1)
        i += 1
        print('--------------The time it takes to process all datasets are {}s --------------'.format(total_time))
    print('--------------FPS = {} --------------'.format(total_frames/total_time))

    # for evaluation
    if args.eval:
        import subprocess
        import pandas as pd
        from tqdm import tqdm
        import re
        import shutil

        src = args.save_path
        dst = args.save_path + '/trackeval'

        os.makedirs(f'{dst}/gt/label_02', exist_ok=True)
        os.makedirs(f'{dst}/trackers/label_02', exist_ok=True)

        scenes = sorted(os.listdir(f'{src}/data'))
        dp = pd.read_csv(args.gt_data, index_col=0, dtype={'frame':object})

        # make gt, evaluate_tracking.seqmap.training
        seqmap = [f'{scene.replace(".txt", "")} empty 0000 0010' for scene in scenes]
        seqmap[0] = re.sub('0000', '0001', seqmap[0])
        with open(f'{dst}/gt/evaluate_tracking.seqmap.training', 'w') as f:
            f.write('\n'.join(seqmap))

        for scene in tqdm(scenes):
            scene = scene.replace('.txt', '')
            
            # make gt, label_02
            scene_df = dp.loc[dp['scene']==scene].copy()
            scene_df[['truncated', 'occluded', 'alpha']]= 0
            scene_df['frame'] = scene_df['frame'].astype(int)
            scene_df[['frame']]
            scene_df = scene_df[['frame', 'id', 'class', 'truncated', 'occluded', 'alpha', 'min_x', 'min_y', 'max_x', 'max_y', 'h', 'w', 'l', 'point_x', 'point_y', 'point_z', 'rot_y']]
            dropped_duple_idx = scene_df[['frame', 'id']].drop_duplicates().index
            scene_df = scene_df.loc[dropped_duple_idx].copy()
            scene_df.to_csv(f'{dst}/gt/label_02/{scene}.txt', index=None, header=None, sep=' ')


            # make trakers, label_02
            if os.path.isfile(f'{src}/data/{scene}.txt')==False:
                null_data = '0 0 None 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
                with open(f'{dst}/trackers/label_02/{scene}.txt', 'w') as f:
                    f.write(null_data)
            else:
                shutil.copy(f'{src}/data/{scene}.txt', f'{dst}/trackers/label_02/{scene}.txt')

        os.chdir('./TrackEval_aivill')
        GT_FOLDER = dst + '/gt'
        TRACKERS_FOLDER = dst + '/trackers'
        OUTPUT_FOLDER = dst
        run_trackeval = f'python scripts/run_kitti.py --GT_FOLDER {GT_FOLDER} --TRACKERS_FOLDER {TRACKERS_FOLDER} --OUTPUT_FOLDER {OUTPUT_FOLDER}'

        subprocess.call(run_trackeval, shell=True)