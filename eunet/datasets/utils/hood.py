import pickle
from typing import Dict

import torch
from smplx import SMPL

from .hood_garment_smpl import GarmentSMPL

from mmcv import Config
import os


def convert_lbs_dict(lbs_dict: dict) -> Dict[str, torch.FloatTensor]:
    """ Convert the values of the lbs_dict from np.array to torch.Tensor"""

    for k in ['shapedirs', 'posedirs', 'lbs_weights', 'v']:
        lbs_dict[k] = torch.FloatTensor(lbs_dict[k])

    return lbs_dict

def load_garments_dict(path: str) -> Dict[str, dict]:
    """ Load the garments_dict containing data for all garments from a pickle file"""

    with open(path, 'rb') as f:
        garments_dict = pickle.load(f)

    for garment, g_dict in garments_dict.items():
        g_dict['lbs'] = convert_lbs_dict(g_dict['lbs'])

    return garments_dict

def make_garment_smpl_dict(garments_dict: Dict[str, dict], smpl_model: SMPL) -> Dict[str, GarmentSMPL]:
    """ For each garment create a GarmentSMPL object"""
    garment_smpl_model_dict = dict()
    for garment, g_dict in garments_dict.items():
        g_smpl_model = GarmentSMPL(smpl_model, g_dict['lbs'])
        garment_smpl_model_dict[garment] = g_smpl_model

    return garment_smpl_model_dict

def make_obstacle_dict(mcfg: Config) -> dict:
    if mcfg.get('obstacle_dict_file', None) is None:
        # Currently is None
        ## This one is to identify the nodetype of the human bodies
        return {}

    obstacle_dict_path = os.path.join(mcfg.aux_data, mcfg.obstacle_dict_file)
    with open(obstacle_dict_path, 'rb') as f:
        obstacle_dict = pickle.load(f)
    return obstacle_dict