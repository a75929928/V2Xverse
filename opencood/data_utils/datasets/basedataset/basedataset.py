# -*- coding: utf-8 -*-
# Initial Author: Runsheng Xu <rxx3386@ucla.edu>
# Revised Author: Qian Huang <huangq@zhejianglab.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Basedataset class for all kinds of fusion.
Import from Select2Col for time delay calculation
"""

import os
import math
from collections import OrderedDict

import torch
import numpy as np
from torch.utils.data import Dataset

import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
# from opencood.hypes_yaml.yaml_utils import _load_json
from opencood.utils.pcd_utils import downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2
import random

import json
import logging
_logger = logging.getLogger(__name__)

import re
import copy

class BaseDataset(Dataset):
    """
    Only reserve time_delay calculation
    
    Base dataset for all kinds of fusion. Mainly used to initialize the
    database and associate the __get_item__ index with the correct timestamp
    and scenario.

    Parameters
    __________
    params : dict
        The dictionary contains all parameters for training/testing.

    visualize : false
        If set to true, the raw point cloud will be saved in the memory
        for visualization.

    Attributes
    ----------
    scenario_database : OrderedDict
        A structured dictionary contains all file information.

    len_record : list
        The list to record each scenario's data length. This is used to
        retrieve the correct index during training.

    pre_processor : opencood.pre_processor
        Used to preprocess the raw data.

    post_processor : opencood.post_processor
        Used to generate training labels and convert the model outputs to
        bbx formats.

    data_augmentor : opencood.data_augmentor
        Used to augment data.

    """

    def __init__(self, params, visualize, train=True, dataset_split_id=-1):
        # if the training/testing include noisy setting
        if 'wild_setting' in params:
            self.seed = params['wild_setting']['seed']
            # whether to add time delay
            self.async_flag = params['wild_setting']['async']
            self.async_mode = \
                'sim' if 'async_mode' not in params['wild_setting'] \
                    else params['wild_setting']['async_mode']
            

            # localization error
            self.loc_err_flag = params['wild_setting']['loc_err']
            self.xyz_noise_std = params['wild_setting']['xyz_std']
            self.ryp_noise_std = params['wild_setting']['ryp_std']

            # transmission data size
            self.data_size = \
                params['wild_setting']['data_size'] \
                    if 'data_size' in params['wild_setting'] else 0
            self.transmission_speed = \
                params['wild_setting']['transmission_speed'] \
                    if 'transmission_speed' in params['wild_setting'] else 27
            self.backbone_delay = \
                params['wild_setting']['backbone_delay'] \
                    if 'backbone_delay' in params['wild_setting'] else 0

        else:
            self.async_flag = False
            self.async_overhead = 0  # ms
            self.async_mode = 'sim'
            self.loc_err_flag = False
            self.xyz_noise_std = 0
            self.ryp_noise_std = 0
            self.data_size = 0  # Mb (Megabits)
            self.transmission_speed = 27  # Mbps
            self.backbone_delay = 0  # ms

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        pass

    def retrieve_base_data(self, idx, cur_ego_pose_flag=True, uni_time_delay=-1, model_type='Select2Col'):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        cur_ego_pose_flag : bool
            Indicate whether to use current timestamp ego pose to calculate
            transformation matrix. If set to false, meaning when other cavs
            project their LiDAR point cloud to ego, they are projecting to
            past ego pose.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]
        # retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database,
                                                  timestamp_index)

        # calculate distance to ego for each cav
        ego_cav_content = \
            self.calc_dist_to_ego(scenario_database, timestamp_key)

        data = OrderedDict()
        # load files for all CAVs
        cavs_num = len(scenario_database.items())
        append_col = 0
        for cav_id, cav_content in scenario_database.items():
            ## perpare parameters for timestamp_delay
            cur_params = self._load_json(cav_content['yaml'])
            cur_ego_params = self._load_json(ego_cav_content['yaml'])

            cur_ego_lidar_pose = cur_ego_params['lidar_pose']
            cur_cav_lidar_pose = cur_params['lidar_pose']
            distance = \
                math.sqrt((cur_cav_lidar_pose[0] -
                           cur_ego_lidar_pose[0]) ** 2 + (
                                  cur_cav_lidar_pose[1] - cur_ego_lidar_pose[
                              1]) ** 2)
            # calculate delay for this vehicle
            timestamp_delay = \
                self.time_delay_calculation(cav_content['ego'], cavs_num, distance, -1)
            # if model_type == 'Select2Col' and not self.train and timestamp_delay>=3:
            #     continue
            if timestamp_index - timestamp_delay <= 0:
                timestamp_delay = timestamp_index
            timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
            timestamp_key_delay = self.return_timestamp_key(scenario_database,
                                                            timestamp_index_delay)

            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']
            # add time delay to vehicle parameters
            data[cav_id]['time_delay'] = timestamp_delay
            # load the corresponding data into the dictionary
            data[cav_id]['params'] = self.reform_param(cav_content,
                                                       ego_cav_content,
                                                       timestamp_key, ## time current
                                                       timestamp_key_delay, ## time delay
                                                       cur_ego_pose_flag)
            data[cav_id]['lidar_np'] = \
                pcd_utils.pcd_to_np(cav_content[timestamp_key_delay]['lidar'])
            if model_type == 'Select2Col' and data[cav_id]['ego'] == True:
                for idxadd in [10000,10001]:
                    data[str(int(cav_id)+idxadd)] = OrderedDict()
                    data[str(int(cav_id)+idxadd)]['ego'] = False
                    timestamp_delay = idxadd - 10000 + 1
                    if timestamp_index - timestamp_delay <= 0:
                        timestamp_delay = timestamp_index
                    timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
                    timestamp_key_delay = self.return_timestamp_key(scenario_database,
                                                                    timestamp_index_delay)
                    # add time delay to vehicle parameters
                    data[str(int(cav_id)+idxadd)]['time_delay'] = timestamp_delay
                    # load the corresponding data into the dictionary

                    data[str(int(cav_id)+idxadd)]['params'] = self.reform_param(cav_content,
                                                               ego_cav_content,
                                                               timestamp_key,  ## time current
                                                               timestamp_key_delay,  ## time delay
                                                               cur_ego_pose_flag)
                    data[str(int(cav_id)+idxadd)]['lidar_np'] = \
                        pcd_utils.pcd_to_np(cav_content[timestamp_key_delay]['lidar'])

        return data
    
    def retrieve_delay_data(self, data, scene_dict, frame_id, visible_actors, tpe, cur_ego_pose_flag=True, uni_time_delay=-1, model_type='Select2Col'):
        """
        Instead of storing a large dataset directory and fetch data with certain index
        Locate its index and read the data directly while training
        """
        scenario_database = data 
        # NOTE idx = data['car_0']['frame_id']?
        timestamp_index = frame_id
        timestamp_key = timestamp_index
        cavs_num = len(scenario_database) 
        ego_cav_content = data['car_0']
        cur_ego_lidar_pose = ego_cav_content['params']['lidar_pose'] 
        
        for cav_id, cav_content in list(scenario_database.items()):
            # Determine type of cav 
            if cav_id == ('car_0'): cav_type = 'ego'
            elif cav_id.startswith('car'): cav_type = 'other_ego'
            elif cav_id.startswith('rsu'): cav_type = 'rsu'

            if cav_type == 'rsu': continue # rsu doesn't exists at first, delay calculation would meet error
            
            cur_cav_lidar_pose = cav_content['params']['lidar_pose']

            distance = \
                math.sqrt((cur_cav_lidar_pose[0] -
                           cur_ego_lidar_pose[0]) ** 2 + (
                                  cur_cav_lidar_pose[1] - cur_ego_lidar_pose[
                              1]) ** 2)
            # calculate delay for this vehicle
            timestamp_delay = \
                self.time_delay_calculation(cav_content['ego'], cavs_num, distance, -1)

            timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
            timestamp_key_delay = timestamp_index_delay 
            # Calculate time_delay for each background vehicle
            # NOTE rsu is ignored for now

            try:
                delay_cav_content = self.get_one_record_v2xverse(data[cav_id]['route_dir'], timestamp_key_delay , agent=cav_type, visible_actors=visible_actors[cav_id], tpe=tpe)
                data[cav_id] = delay_cav_content
                data[cav_id]['time_delay'] = timestamp_delay

            except Exception as e:
                print(f"Value Error: {e}")
            

            if model_type == 'Select2Col' and data[cav_id]['ego'] == True:
                for idxadd in [10000,10001]:
                    cav_delay_id = 'car'+ str(int(cav_id.split('_')[-1])+idxadd)
                    # cav_delay_id = str(int(cav_id)+idxadd)
                    data[cav_delay_id] = OrderedDict()
                    timestamp_delay = idxadd - 10000 + 1
                    timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
                    timestamp_key_delay = timestamp_index_delay

                    data[cav_delay_id]['time_delay'] = timestamp_delay

                    # NOTE consider storage path of different ego agents
                    # delay_cav_content = self.get_one_record_v2xverse(data[cav_id]['route_dir'], timestamp_key_delay , agent=cav_type, visible_actors=visible_actors[cav_id], tpe=tpe)
                    # delay_params = delay_cav_content['params']
                    delay_ego_content = self.get_one_record_v2xverse(scene_dict['ego'], timestamp_key_delay , agent='ego', visible_actors=visible_actors['car_0'], tpe=tpe)
                    
                    # NOTE only params and lidar_np needed, get all attributes for convenience
                    data[cav_delay_id] = delay_ego_content
                    data[cav_delay_id]['ego'] = False
        return data
    
    def retrieve_delay_data_closed_loop(self, data, tpe):
        """
        data deemed to have ample historical information about ego and rsu
        ego gets additional historical information
        rsu only get delayed historical information
        """
        # e.g record 6 frames of data, real time delay stored in ['time_delay']
        # {'car_0': ...; 'car_1':...; 'rsu_0': ...; 'rsu_6':...} 
        # car_0 is ALWAYS current ego while others should be past ego
        # rsu_0 and several others could be current or past rsu
        processed_data = {}

        # Empty historical data at first
        if 'time_delay' not in data['car_0']:
            processed_data['car_0'] = copy.copy(data['car_0']) # initially only get current data
            processed_data['car_0']['time_delay'] = 0
            return processed_data
        
        time_delay_database = {}
        # Organize base data according to time delay
        for cav_id, cav_content in data.items():
            if cav_content['time_delay'] not in time_delay_database:
                time_delay_database.update({cav_content['time_delay']: {cav_id: cav_content}})
            else:
                time_delay_database[cav_content['time_delay']].update({cav_id: cav_content})

        # Determine current cav nums
        # NOTE changing RSU would influce its accuracy
        cavs_num = len(time_delay_database[0].keys())

        # car_1, ..., car_n is historical data here
        ego_cav_content = data['car_0']
        cur_ego_lidar_pose = ego_cav_content['params']['lidar_pose'] 

        # Calculate current distance from ego to rsu, and then determine delayed information
        for cav_id, cav_content in list(time_delay_database[0].items()):

            if cav_id == ('car_0'): cav_type = 'ego'
            elif cav_id.startswith('rsu'): cav_type = 'rsu'
            
            cur_cav_lidar_pose = cav_content['params']['lidar_pose']

            distance = \
                math.sqrt((cur_cav_lidar_pose[0] -
                        cur_ego_lidar_pose[0]) ** 2 + (
                                cur_cav_lidar_pose[1] - cur_ego_lidar_pose[
                            1]) ** 2)
            # calculate delay for this vehicle
            timestamp_delay = \
                self.time_delay_calculation(cav_content['ego'], cavs_num, distance, -1)
            
            # find nearest delay data in buffer
            timestamp_key_delay = find_closest_simple(time_delay_database.keys(), timestamp_delay) 

            try:
                delay_cav_content = time_delay_database[timestamp_key_delay][cav_id]
                processed_data[cav_id] = delay_cav_content
                processed_data[cav_id]['time_delay'] = timestamp_key_delay
                # delay_cav_content = self.get_one_record_v2xverse(data[cav_id]['route_dir'], timestamp_key_delay , agent=cav_type, visible_actors=visible_actors[cav_id], tpe=tpe)
            except Exception as e:
                print(f"Value Error: {e}")

        ## historical data, stored seperately as background ego
        for idxadd in [10000,10001]:
            cav_delay_id = 'car'+ str(idxadd)
            # cav_delay_id = 'car'+ str(int(cav_id.split('_')[-1])+idxadd)
            # cav_delay_id = str(int(cav_id)+idxadd)
            processed_data[cav_delay_id] = OrderedDict()
            timestamp_delay = idxadd - 10000 + 1

            # NOTE consider storage path of different ego agents
            # delay_cav_content = self.get_one_record_v2xverse(processed_data[cav_id]['route_dir'], timestamp_key_delay , agent=cav_type, visible_actors=visible_actors[cav_id], tpe=tpe)
            # delay_params = delay_cav_content['params']
            if 'time_delay' in data['car_0']:
                # batch causes interval change time_delay_database [0, 1, 2, 3, 4] --> [0, 4, 5, 6, 7]
                # so simply using closest timestamp may cause time delay always equal to 0
                processed_data[cav_delay_id]['time_delay'] = find_closest_simple(time_delay_database.keys(), timestamp_delay) 
                timestamp_index_delay = processed_data[cav_delay_id]['time_delay'] 
                # find past ego data, since only ego is car, simply use startswith as filter
                for cav_id, cav_content in time_delay_database[timestamp_index_delay].items():
                    if cav_id.startswith('car'):
                        delay_ego_content = copy.copy(cav_content)
                        break
                # else:
                #     delay_ego_content = copy.copy(data['car_0']) # or attribute 'ego' would affect car_0
                #     delay_ego_content['time_delay'] = timestamp_delay
                # delay_ego_content = self.get_one_record_v2xverse(scene_dict['ego'], timestamp_key_delay , agent='ego', visible_actors=visible_actors['car_0'], tpe=tpe)
            
                # NOTE only params and lidar_np needed, get all attributes for convenience
                processed_data[cav_delay_id] = delay_ego_content
                processed_data[cav_delay_id]['ego'] = False

        return processed_data

    
    @staticmethod
    def extract_timestamps(json_files):
        """
        Given the list of the json files, extract the mocked timestamps.

        Parameters
        ----------
        json_files : list
            The full path of all json files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in json_files:
            res = file.split('/')[-1]

            timestamp = res.replace('.json', '')
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # get all timestamp keys
        timestamp_keys = list(scenario_database.items())[0][1]
        # retrieve the correct index
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

        return timestamp_key
    def calc_dist_to_ego(self, scenario_database, timestamp_key):
        """
        Calculate the distance to ego for each cav.
        """
        ego_lidar_pose = None
        ego_cav_content = None
        # Find ego pose first
        for cav_id, cav_content in scenario_database.items():
            if cav_content['ego']:
                ego_cav_content = cav_content
                ego_lidar_pose = \
                    self._load_json(cav_content['json'])['lidar_pose']
                break

        assert ego_lidar_pose is not None

        # calculate the distance
        for cav_id, cav_content in scenario_database.items():
            cur_lidar_pose = \
                self._load_json(cav_content['json'])['lidar_pose']
            distance = \
                math.sqrt((cur_lidar_pose[0] -
                           ego_lidar_pose[0]) ** 2 +
                          (cur_lidar_pose[1] - ego_lidar_pose[1]) ** 2)
            cav_content['distance_to_ego'] = distance
            scenario_database.update({cav_id: cav_content})

        return ego_cav_content

    def trans_time(self, bandwidth, cavs_num, distance2ego):
        F = 1.06
        B = bandwidth / cavs_num
        P = 23
        if distance2ego < 1:
            distance2ego = 1
        PL = 28 + 22 * math.log10(distance2ego) + 20 * math.log10(5.9)
        N = -1 * random.randint(95, 110)
        PPN = P - PL - N
        T = F / (B * math.log2(1 + math.pow(10, 0.1 * PPN))) * 1000
        return T
    
    def time_delay_calculation(self, ego_flag, cavs_num, distance2ego, uni_time_delay):
        """
        Calculate the time delay for a certain vehicle.

        Parameters
        ----------
        ego_flag : boolean
            Whether the current cav is ego.

        Return
        ------
        time_delay : int
            The time delay quantization.
        """
        # there is not time delay for ego vehicle
        if ego_flag:
            return 0
        if uni_time_delay>=0:
            return uni_time_delay
        # time delay real mode
        elif self.async_mode == 'iosi':
            distance2ego = distance2ego
            T_trans_time = self.trans_time(10, cavs_num, distance2ego)
            T_sensor_time = random.randint(0, 100)
            # based on real device
            T_compute_time = random.randint(20, 40)
            ## T_other_time indicate system other overhead, for instance, read sensor data overhead, waiting time
            T_other_time = random.randint(0, 100)
            T_time_delay = T_trans_time + T_sensor_time + T_compute_time + T_other_time
            time_delay = int(T_time_delay)
        time_delay = time_delay // 100
        return time_delay if self.async_flag else 0

    def add_loc_noise(self, pose, xyz_std, ryp_std):
        """
        Add localization noise to the pose.

        Parameters
        ----------
        pose : list
            x,y,z,roll,yaw,pitch

        xyz_std : float
            std of the gaussian noise on xyz

        ryp_std : float
            std of the gaussian noise
        """
        np.random.seed(self.seed)
        xyz_noise = np.random.normal(0, xyz_std, 3)
        ryp_std = np.random.normal(0, ryp_std, 3)
        noise_pose = [pose[0] + xyz_noise[0],
                      pose[1] + xyz_noise[1],
                      pose[2] + xyz_noise[2],
                      pose[3],
                      pose[4] + ryp_std[1],
                      pose[5]]
        return noise_pose
    
    def reform_param_v2xverse(self, cav_content, ego_content, timestamp_cur,
                     timestamp_delay, cur_ego_pose_flag,
                     delay_params, delay_ego_params): # added params
        """
        Reform the data params with current timestamp object groundtruth and
        delay timestamp LiDAR pose for other CAVs.
        """
        cur_params = cav_content['params']
        cur_ego_params = ego_content['params']
        # at the delayed timestamp
        delay_cav_lidar_pose = delay_params['lidar_pose']
        delay_ego_lidar_pose = delay_ego_params["lidar_pose"]

        cur_ego_lidar_pose = cur_ego_params['lidar_pose']
        cur_cav_lidar_pose = cur_params['lidar_pose']
        if cur_ego_pose_flag:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose,
                                             cur_ego_lidar_pose)
            spatial_correction_matrix = np.eye(4)
        else:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose,
                                             delay_ego_lidar_pose)
            spatial_correction_matrix = x1_to_x2(delay_ego_lidar_pose,
                                                 cur_ego_lidar_pose)
        # This is only used for late fusion, as it did the transformation
        # in the postprocess, so we want the gt object transformation use
        # the correct one
        gt_transformation_matrix = x1_to_x2(cur_cav_lidar_pose,
                                            cur_ego_lidar_pose)

        # we always use current timestamp's gt bbx to gain a fair evaluation
        delay_params['vehicles'] = cur_params['vehicles']
        delay_params['transformation_matrix'] = transformation_matrix
        delay_params['gt_transformation_matrix'] = \
            gt_transformation_matrix
        delay_params['spatial_correction_matrix'] = spatial_correction_matrix

        return delay_params
    
    def project_points_to_bev_map(self, points, ratio=0.1):
        """
        Project points to BEV occupancy map with default ratio=0.1.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) / (N, 4)

        ratio : float
            Discretization parameters. Default is 0.1.

        Returns
        -------
        bev_map : np.ndarray
            BEV occupancy map including projected points
            with shape (img_row, img_col).

        """
        return self.pre_processor.project_points_to_bev_map(points, ratio)

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.

        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape

        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw

        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask
    def visualize_result(self, pred_box_tensor,
                         gt_tensor,
                         pcd,
                         show_vis,
                         save_path,
                         dataset=None):
        # visualize the model output
        self.post_processor.visualize(pred_box_tensor,
                                      gt_tensor,
                                      pcd,
                                      show_vis,
                                      save_path,
                                      dataset=dataset)
        
    def _load_json(self, path):
        try:
            if path.startswith(self.root_dir):
                json_value = json.load(open(path))
            else:
                json_value = json.load(open(os.path.join(self.root_dir,path)))
        except Exception as e:
            _logger.info(path)
            n = path[-9:-5]
            new_path = path[:-9] + "%04d.json" % (int(n) - 1)
            json_value = json.load(open(os.path.join(self.root_dir,new_path)))
        return json_value
    
    def get_one_record_v2xverse(self, route_dir, frame_id, agent='ego', visible_actors=None, tpe='all', extra_source=None):
        output_record = OrderedDict()

        if agent == 'ego':
            output_record['ego'] = True
        else:
            output_record['ego'] = False

        BEV = None

        if route_dir is not None:
            measurements = self._load_json(os.path.join(route_dir, "measurements", "%04d.json" % frame_id))
            actors_data = self._load_json(os.path.join(route_dir, "actors_data", "%04d.json" % frame_id))
        elif extra_source is not None:
            if 'actors_data' in extra_source:
                actors_data = extra_source['actors_data']
            else:
                actors_data = {}
            measurements = extra_source['measurements']

        ego_loc = np.array([measurements['x'], measurements['y']])
        output_record['params'] = {}
        
        cam_list = ['front','right','left','rear']
        cam_angle_list = [0, 60, -60, 180]
        for cam_id in range(4):
            output_record['params']['camera{}'.format(cam_id)] = {}
            output_record['params']['camera{}'.format(cam_id)]['cords'] = [measurements['x'], measurements['y'], 1.0,\
	 						                                                0,measurements['theta']/np.pi*180+cam_angle_list[cam_id],0]
            output_record['params']['camera{}'.format(cam_id)]['extrinsic'] = measurements['camera_{}_extrinsics'.format(cam_list[cam_id])]
            output_record['params']['camera{}'.format(cam_id)]['intrinsic'] = measurements['camera_{}_intrinsics'.format(cam_list[cam_id])]

        if 'speed' in measurements:
            output_record['params']['ego_speed'] = measurements['speed']*3.6
        else:
            output_record['params']['ego_speed'] = 0

        output_record['params']['lidar_pose'] = \
                        [measurements['lidar_pose_x'], measurements['lidar_pose_y'], 0, \
                        0,measurements['theta']/np.pi*180-90,0]
        self.distance_to_map_center = (self.det_range[3]-self.det_range[0])/2+self.det_range[0]
        output_record['params']['map_pose'] = \
                        [measurements['lidar_pose_x'] + self.distance_to_map_center*np.cos(measurements["theta"]-np.pi/2),
                         measurements['lidar_pose_y'] + self.distance_to_map_center*np.sin(measurements["theta"]-np.pi/2), 0, \
                        0,measurements['theta']/np.pi*180-90,0]
        detmap_pose_x = measurements['lidar_pose_x'] + self.distance_to_map_center*np.cos(measurements["theta"]-np.pi/2)
        detmap_pose_y = measurements['lidar_pose_y'] + self.distance_to_map_center*np.sin(measurements["theta"]-np.pi/2)
        detmap_theta = measurements["theta"] + np.pi/2
        output_record['detmap_pose'] = np.array([-detmap_pose_y, detmap_pose_x, detmap_theta])
        output_record['params']['lidar_pose_clean'] = output_record['params']['lidar_pose']
        output_record['params']['plan_trajectory'] = []
        output_record['params']['true_ego_pos'] = \
                        [measurements['lidar_pose_x'], measurements['lidar_pose_y'], 0, \
                         0,measurements['theta']/np.pi*180,0]
        output_record['params']['predicted_ego_pos'] = \
                        [measurements['lidar_pose_x'], measurements['lidar_pose_y'], 0, \
                        0,measurements['theta']/np.pi*180,0]
        
        if True: # or it would cause lidar_np missing
        # if tpe == 'all':
            if route_dir is not None:
                lidar = self._load_npy(os.path.join(route_dir, "lidar", "%04d.npy" % frame_id))
                output_record['rgb_front'] = self._load_image(os.path.join(route_dir, "rgb_front", "%04d.jpg" % frame_id))
                output_record['rgb_left'] = self._load_image(os.path.join(route_dir, "rgb_left", "%04d.jpg" % frame_id))
                output_record['rgb_right'] = self._load_image(os.path.join(route_dir, "rgb_right", "%04d.jpg" % frame_id))
                output_record['rgb_rear'] = self._load_image(os.path.join(route_dir, "rgb_rear", "%04d.jpg" % frame_id))
                if agent != 'rsu':
                    BEV = self._load_image(os.path.join(route_dir, "birdview", "%04d.jpg" % frame_id))
            elif extra_source is not None:
                lidar = extra_source['lidar']
                if 'rgb_front' in extra_source:
                    output_record['rgb_front'] = extra_source['rgb_front']
                    output_record['rgb_left'] = extra_source['rgb_left']
                    output_record['rgb_right'] = extra_source['rgb_right']
                    output_record['rgb_rear'] = extra_source['rgb_rear']
                else:
                    output_record['rgb_front'] = None
                    output_record['rgb_left'] = None
                    output_record['rgb_right'] = None
                    output_record['rgb_rear'] = None
                BEV = None

            output_record['lidar_np'] = lidar
            lidar_transformed = np.zeros((output_record['lidar_np'].shape))
            lidar_transformed[:,0] = output_record['lidar_np'][:,1]
            lidar_transformed[:,1] = -output_record['lidar_np'][:,0]
            lidar_transformed[:,2:] = output_record['lidar_np'][:,2:]
            output_record['lidar_np'] = lidar_transformed.astype(np.float32)
            output_record['lidar_np'][:, 2] += measurements['lidar_pose_z']

        if visible_actors is not None:
            actors_data = self.filter_actors_data_according_to_visible(actors_data, visible_actors)

        ################ LSS debug TODO: clean up this function #####################
        if not self.first_det:
            import copy
            if True: # agent=='rsu':
                measurements["affected_light_id"] = -1
                measurements["is_vehicle_present"] = []
                measurements["is_bike_present"] = []
                measurements["is_junction_vehicle_present"] = []
                measurements["is_pedestrian_present"] = []
                measurements["future_waypoints"] = []
            cop3_range = [36,12,12,12, 0.25]
            heatmap = generate_heatmap_multiclass(
                copy.deepcopy(measurements), copy.deepcopy(actors_data), max_distance=36
            )
            self.det_data = (
                generate_det_data_multiclass(
                    heatmap, copy.deepcopy(measurements), copy.deepcopy(actors_data), cop3_range
                )
                .reshape(3, int((cop3_range[0]+cop3_range[1])/cop3_range[4]
                            *(cop3_range[2]+cop3_range[3])/cop3_range[4]), -1) #(2, H*W,7)
                .astype(np.float32)
            )
            self.first_det = True
            if self.label_mode == 'cop3':
                self.first_det = False
        output_record['det_data'] = self.det_data
        ##############################################################
        if agent == 'rsu' :
            for actor_id in actors_data.keys():
                if actors_data[actor_id]['tpe'] == 0:
                    box = actors_data[actor_id]['box']
                    if abs(box[0]-0.8214) < 0.01 and abs(box[1]-0.18625) < 0.01 :
                        actors_data[actor_id]['tpe'] = 3

        output_record['params']['vehicles'] = {}
        for actor_id in actors_data.keys():

            ######################
            ## debug
            ######################
            # if agent == 'ego':
            #     continue

            if tpe in [0, 1, 3]:
                if actors_data[actor_id]['tpe'] != tpe:
                    continue

            # exclude ego car
            loc_actor = np.array(actors_data[actor_id]['loc'][0:2])
            dis = np.linalg.norm(ego_loc - loc_actor)
            if dis < 0.1:
                continue

            if not ('box' in actors_data[actor_id].keys() and 'ori' in actors_data[actor_id].keys() and 'loc' in actors_data[actor_id].keys()):
                continue
            output_record['params']['vehicles'][actor_id] = {}
            output_record['params']['vehicles'][actor_id]['tpe'] = actors_data[actor_id]['tpe']
            yaw = math.degrees(math.atan(actors_data[actor_id]['ori'][1]/actors_data[actor_id]['ori'][0]))
            pitch = math.degrees(math.asin(actors_data[actor_id]['ori'][2]))
            output_record['params']['vehicles'][actor_id]['angle'] = [0,yaw,pitch]
            output_record['params']['vehicles'][actor_id]['center'] = [0,0,actors_data[actor_id]['box'][2]]
            output_record['params']['vehicles'][actor_id]['extent'] = actors_data[actor_id]['box']
            output_record['params']['vehicles'][actor_id]['location'] = [actors_data[actor_id]['loc'][0],actors_data[actor_id]['loc'][1],0]
            output_record['params']['vehicles'][actor_id]['speed'] = 3.6 * math.sqrt(actors_data[actor_id]['vel'][0]**2+actors_data[actor_id]['vel'][1]**2 )

        direction_list = ['front','left','right','rear']
        theta_list = [0,-60,60,180]
        dis_list = [0,0,0,-2.6]
        camera_data_list = []
        for i, direction in enumerate(direction_list):
            if 'rgb_{}'.format(direction) in output_record:
                camera_data_list.append(output_record['rgb_{}'.format(direction)])
            dis_to_lidar = dis_list[i]
            output_record['params']['camera{}'.format(i)]['cords'] = \
                                                                    [measurements['x'] + dis_to_lidar*np.sin(measurements['theta']), measurements['y'] - dis_to_lidar*np.cos(measurements['theta']), 2.3,\
                                                                    0,measurements['theta']/np.pi*180 - 90  + theta_list[i],0]
            output_record['params']['camera{}'.format(i)]['extrinsic'] = measurements['camera_{}_extrinsics'.format(direction_list[i])]
            output_record['params']['camera{}'.format(i)]['intrinsic'] = measurements['camera_{}_intrinsics'.format(direction_list[i])]
        
        output_record['camera_data'] = camera_data_list
        bev_visibility_np = 255*np.ones((256,256,3), dtype=np.uint8)
        output_record['bev_visibility.png'] = bev_visibility_np

        if agent != 'rsu':
            output_record['BEV'] = BEV
        else:
            output_record['BEV'] = None

        return output_record
    
    def generate_object_center_lidar(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_object_center(cav_contents,
                                                        reference_lidar_pose)

VALUES = [255]
EXTENT = [0]
def generate_heatmap_multiclass(measurements, actors_data, max_distance=30, pixels_per_meter=8):
    actors_data_multiclass = {
        0: {}, 1: {}, 2:{}, 3:{}
    }
    for _id in actors_data.keys():
        actors_data_multiclass[actors_data[_id]['tpe']][_id] = actors_data[_id]
    heatmap_0 = generate_heatmap(measurements, actors_data_multiclass[0], max_distance, pixels_per_meter)
    heatmap_1 = generate_heatmap(measurements, actors_data_multiclass[1], max_distance, pixels_per_meter)
    # heatmap_2 = generate_heatmap(measurements, actors_data_multiclass[2], max_distance, pixels_per_meter) # traffic light, not used
    heatmap_3 = generate_heatmap(measurements, actors_data_multiclass[3], max_distance, pixels_per_meter)
    return {0: heatmap_0, 1: heatmap_1, 3: heatmap_3}

def get_yaw_angle(forward_vector):
    forward_vector = forward_vector / np.linalg.norm(forward_vector)
    yaw = math.acos(forward_vector[0])
    if forward_vector[1] < 0:
        yaw = 2 * np.pi - yaw
    return yaw

def generate_heatmap(measurements, actors_data, max_distance=30, pixels_per_meter=8):
    img_size = max_distance * pixels_per_meter * 2
    img = np.zeros((img_size, img_size, 3), np.int)
    ego_x = measurements["lidar_pose_x"]
    ego_y = measurements["lidar_pose_y"]
    ego_theta = measurements["theta"]
    R = np.array(
        [
            [np.cos(ego_theta), -np.sin(ego_theta)],
            [np.sin(ego_theta), np.cos(ego_theta)],
        ]
    )
    ego_id = None
    for _id in actors_data:
        color = np.array([1, 1, 1])
        if actors_data[_id]["tpe"] == 2:
            if int(_id) == int(measurements["affected_light_id"]):
                if actors_data[_id]["sta"] == 0:
                    color = np.array([1, 1, 1])
                else:
                    color = np.array([0, 0, 0])
                yaw = get_yaw_angle(actors_data[_id]["ori"])
                TR = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
                actors_data[_id]["loc"] = np.array(
                    actors_data[_id]["loc"][:2]
                ) + TR.T.dot(np.array(actors_data[_id]["taigger_loc"])[:2])
                actors_data[_id]["ori"] = np.array(actors_data[_id]["ori"])
                actors_data[_id]["box"] = np.array(actors_data[_id]["trigger_box"]) * 2
            else:
                continue
        raw_loc = actors_data[_id]["loc"]
        if (raw_loc[0] - ego_x) ** 2 + (raw_loc[1] - ego_y) ** 2 <= 2:
            ego_id = _id
            color = np.array([0, 1, 1])
        new_loc = R.T.dot(np.array([raw_loc[0] - ego_x, raw_loc[1] - ego_y]))
        actors_data[_id]["loc"] = np.array(new_loc)
        raw_ori = actors_data[_id]["ori"]
        new_ori = R.T.dot(np.array([raw_ori[0], raw_ori[1]]))
        actors_data[_id]["ori"] = np.array(new_ori)
        actors_data[_id]["box"] = np.array(actors_data[_id]["box"])
        if int(_id) in measurements["is_vehicle_present"]:
            color = np.array([1, 1, 1])
        elif int(_id) in measurements["is_bike_present"]:
            color = np.array([1, 1, 1])
        elif int(_id) in measurements["is_junction_vehicle_present"]:
            color = np.array([1, 1, 1])
        elif int(_id) in measurements["is_pedestrian_present"]:
            color = np.array([1, 1, 1])
        actors_data[_id]["color"] = color

    if ego_id is not None and ego_id in actors_data:
        del actors_data[ego_id]  # Do not show ego car
    for _id in actors_data:
        if actors_data[_id]["tpe"] == 2:
            continue  # FIXME donot add traffix light
            if int(_id) != int(measurements["affected_light_id"]):
                continue
            if actors_data[_id]["sta"] != 0:
                continue
        act_img = np.zeros((img_size, img_size, 3), np.uint8)
        loc = actors_data[_id]["loc"][:2]
        ori = actors_data[_id]["ori"][:2]
        box = actors_data[_id]["box"]
        if box[0] < 1.5:
            box = box * 1.5  # FIXME enlarge the size of pedstrian and bike
        color = actors_data[_id]["color"]
        for i in range(len(VALUES)):
            act_img = add_rect(
                act_img,
                loc,
                ori,
                box + EXTENT[i],
                VALUES[i],
                pixels_per_meter,
                max_distance,
                color,
            )
        act_img = np.clip(act_img, 0, 255)
        img = img + act_img
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    img = img[:, :, 0]
    return img

def add_rect(img, loc, ori, box, value, pixels_per_meter, max_distance, color):
    img_size = max_distance * pixels_per_meter * 2
    vet_ori = np.array([-ori[1], ori[0]])
    hor_offset = box[0] * ori
    vet_offset = box[1] * vet_ori
    left_up = (loc + hor_offset + vet_offset + max_distance) * pixels_per_meter
    left_down = (loc + hor_offset - vet_offset + max_distance) * pixels_per_meter
    right_up = (loc - hor_offset + vet_offset + max_distance) * pixels_per_meter
    right_down = (loc - hor_offset - vet_offset + max_distance) * pixels_per_meter
    left_up = np.around(left_up).astype(np.int)
    left_down = np.around(left_down).astype(np.int)
    right_down = np.around(right_down).astype(np.int)
    right_up = np.around(right_up).astype(np.int)
    left_up = list(left_up)
    left_down = list(left_down)
    right_up = list(right_up)
    right_down = list(right_down)
    color = [int(x) for x in value * color]
    cv2.fillConvexPoly(img, np.array([left_up, left_down, right_down, right_up]), color)
    return img

def generate_det_data_multiclass(
    heatmap, measurements, actors_data, det_range=[30,10,10,10, 0.8]
):  
    actors_data_multiclass = {
        0: {}, 1: {}, 2: {}, 3:{}
    }
    for _id in actors_data.keys():
        actors_data_multiclass[actors_data[_id]['tpe']][_id] = actors_data[_id]
    det_data = []
    for _class in range(4):
        if _class != 2:
            det_data.append(generate_det_data(heatmap[_class], measurements, actors_data_multiclass[_class], det_range))

    return np.array(det_data)

from skimage.measure import block_reduce

def generate_det_data(
    heatmap, measurements, actors_data, det_range=[30,10,10,10, 0.8]
):
    res = det_range[4]
    max_distance = max(det_range)
    traffic_heatmap = block_reduce(heatmap, block_size=(int(8*res), int(8*res)), func=np.mean)
    traffic_heatmap = np.clip(traffic_heatmap, 0.0, 255.0)
    traffic_heatmap = traffic_heatmap[:int((det_range[0]+det_range[1])/res), int((max_distance-det_range[2])/res):int((max_distance+det_range[3])/res)]
    det_data = np.zeros((int((det_range[0]+det_range[1])/res), int((det_range[2]+det_range[3])/res), 7)) # (50,25,7)
    vertical, horizontal = det_data.shape[:2]

    ego_x = measurements["lidar_pose_x"]
    ego_y = measurements["lidar_pose_y"]
    ego_theta = measurements["theta"]
    R = np.array(
        [
            [np.cos(ego_theta), -np.sin(ego_theta)],
            [np.sin(ego_theta), np.cos(ego_theta)],
        ]
    )
    need_deleted_ids = []
    for _id in actors_data:
        raw_loc = actors_data[_id]["loc"]
        new_loc = R.T.dot(np.array([raw_loc[0] - ego_x, raw_loc[1] - ego_y]))
        new_loc[1] = -new_loc[1]
        actors_data[_id]["loc"] = np.array(new_loc)
        raw_ori = actors_data[_id]["ori"]
        new_ori = R.T.dot(np.array([raw_ori[0], raw_ori[1]]))
        dis = new_loc[0] ** 2 + new_loc[1] ** 2
        if (
            dis <= 2
            or dis >= (max_distance) ** 2 * 2
            or "box" not in actors_data[_id]
            or actors_data[_id]['tpe'] == 2
        ):
            need_deleted_ids.append(_id)
            continue
        actors_data[_id]["ori"] = np.array(new_ori)
        actors_data[_id]["box"] = np.array(actors_data[_id]["box"])

    for _id in need_deleted_ids:
        del actors_data[_id]

    for i in range(vertical):  # 50
        for j in range(horizontal):  # 25
            if traffic_heatmap[i][j] < 0.05 * 255.0:
                continue
            center_x, center_y = convert_grid_to_xy(i, j, det_range)
            min_dis = 1000
            min_id = None
            for _id in actors_data:
                loc = actors_data[_id]["loc"][:2]
                ori = actors_data[_id]["ori"][:2]
                box = actors_data[_id]["box"]
                dis = (loc[0] - center_x) ** 2 + (loc[1] - center_y) ** 2
                if dis < min_dis:
                    min_dis = dis
                    min_id = _id

            if min_id is None:
                continue

            loc = actors_data[min_id]["loc"][:2]
            ori = actors_data[min_id]["ori"][:2]
            box = actors_data[min_id]["box"]
            theta = (get_yaw_angle(ori) / np.pi + 2) % 2
            speed = np.linalg.norm(actors_data[min_id]["vel"])

            # prob = np.power(0.5 / max(0.5, np.sqrt(min_dis)), 0.5)

            det_data[i][j] = np.array(
                [
                    0,
                    (loc[0] - center_x) * 3.0,
                    (loc[1] - center_y) * 3.0,
                    theta / 2.0,
                    box[0] / 7.0,
                    box[1] / 4.0,
                    0,
                ]
            )

    heatmap = np.zeros((int((det_range[0]+det_range[1])/res), int((det_range[2]+det_range[3])/res))) # (50,25)
    for _id in actors_data:
        loc = actors_data[_id]["loc"][:2]
        ori = actors_data[_id]["ori"][:2]
        box = actors_data[_id]["box"]
        try:
            x,y = loc
            i,j = convert_xy_to_grid(x,y,det_range)
            i = int(np.around(i))
            j = int(np.around(j))

            if i < vertical and i > 0 and j > 0 and j < horizontal:
                det_data[i][j][-1] = 1.0

            ################## Gaussian Heatmap #####################
            w, h = box[:2]/det_range[4]
            heatmap = draw_heatmap(heatmap, h, w, j, i)
            #########################################################

            # theta = (get_yaw_angle(ori) / np.pi + 2) % 2
            # center_x, center_y = convert_grid_to_xy(i, j, det_range)

            # det_data[i][j] = np.array(
            #     [
            #         0,
            #         (loc[0] - center_x) * 3.0,
            #         (loc[1] - center_y) * 3.0,
            #         theta / 2.0,
            #         box[0] / 7.0,
            #         box[1] / 4.0,
            #         0,
            #     ]
            # )

        except:
            print('actor data error, skip!')
    det_data[:,:,0] = heatmap
    return det_data

def convert_grid_to_xy(i, j, det_range):
    x = det_range[4]*(j + 0.5) - det_range[2]
    y = det_range[0] - det_range[4]*(i+0.5)
    return x, y

def convert_xy_to_grid(x, y, det_range):
    j = (x + det_range[2]) / det_range[4] - 0.5
    i = (det_range[0] - y) / det_range[4] - 0.5
    return i, j

def draw_heatmap(heatmap, h, w, x, y):
    feature_map_size = heatmap.shape
    radius = gaussian_radius(
                    (h, w),
                    min_overlap=0.1)
    radius = max(2, int(radius))

    # throw out not in range objects to avoid out of array
    # area when creating the heatmap
    if not (0 <= y < feature_map_size[0]
            and 0 <= x < feature_map_size[1]):
        return heatmap

    heatmap = draw_gaussian(heatmap, (x,y), radius) 
    return heatmap

def draw_gaussian(heatmap, center, radius, k=1):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked.
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gausian.
        K (int): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                                radius - left:radius + right]
    
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        # torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    #     masked_heatmap = np.max([masked_heatmap[None,], (masked_gaussian * k)[None,]], axis=0)[0]
    # heatmap[y - top:y + bottom, x - left:x + right] = masked_heatmap
    return heatmap

def gaussian_2d(shape, sigma=1):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map.
        sigma (float): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian_radius(det_size, min_overlap=0.5):
    """Get radius of gaussian.

    Args:
        det_size (tuple[torch.Tensor]): Size of the detection result.
        min_overlap (float): Gaussian_overlap. Defaults to 0.5.

    Returns:
        torch.Tensor: Computed radius.
    """
    height, width = det_size
    
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)

def find_closest_simple(nums, target):
        closest_val = None
        smallest_diff = float('inf') 

        for idx, num in enumerate(nums):
            diff = abs(target - num)
            if diff < smallest_diff:
                smallest_diff = diff
                closest_val = num

        return closest_val