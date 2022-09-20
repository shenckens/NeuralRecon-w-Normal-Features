import numpy as np
import torch


def collate_fn(list_data):
    # Concatenate all lists

    tsdf_list = list(zip(*[item['tsdf_list'] for item in list_data]))
    for i in range(len(tsdf_list)):
        tsdf_list[i] = torch.stack(tsdf_list[i]).long()

    return {
        'imgs': torch.stack([item['imgs'] for item in list_data]),
        'vol_origin': torch.stack([item['vol_origin'] for item in list_data]),
        'vol_origin_partial': torch.stack([item['vol_origin_partial'] for item in list_data]),
        'world_to_aligned_camera': torch.stack([item['world_to_aligned_camera'] for item in list_data]),
        'proj_matrices': torch.stack([item['proj_matrices'] for item in list_data]),
        'plane_anchors': [item['plane_anchors'] for item in list_data],
        'residual': [item['residual'] for item in list_data],
        'planes': [item['planes'] for item in list_data],
        'planes_trans': [item['planes_trans'] for item in list_data],
        'instance_pointnum': torch.cat([item['instance_pointnum'] for item in list_data]).int(),
        'tsdf_list': tsdf_list,
        'mean_xyz': [item['mean_xyz'] for item in list_data],
        'scene': [item['scene'] for item in list_data],
        'fragment': [item['fragment'] for item in list_data],
        'epoch': torch.tensor(list_data[0]['epoch']),
        'occ_list': [torch.stack([item['occ_list'][i] for item in list_data]) for i in
                     range(len(list_data[0]['occ_list']))],
    }
