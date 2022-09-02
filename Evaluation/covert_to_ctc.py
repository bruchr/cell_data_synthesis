from skimage.measure import regionprops, label
import tifffile as tiff
import numpy as np
import os
import json


'''
Creates the folder structure required for the cell tracking challenge evaluation.
http://celltrackingchallenge.net/evaluation-methodology/
Allows the use of a mask for image cropping.
'''


seg_name = 'DL_Seg_TD_Sim+Opti'
path = {
    'seg': f'./Data/Segmentation_Results/{seg_name}/pp_2_DAPI.tif',
    'gt':   './Data/Segmentation_Results/Center_Point_GT/seg_center_points_2_DAPI.tif',
    'raw':  './Data/Real/2_DAPI.tif',
}

path_out = f'./Data/Segmentation_Results/CTC_Eval/{seg_name}'

use_mask = True
mask_start = (0,243,366)
mask_size = (32,256,256)

######################################################

print(f'Path_out: {path_out}')

json_dict = {
    'path_seg': path['seg'],
    'path_gt': path['gt'],
    'path_raw': path['raw'],
    'mask_start': mask_start,
    'mask_size': mask_size,
}
os.makedirs(path_out, exist_ok=False)
with open(os.path.join(path_out, 'info.json'), 'w') as json_file:
    json.dump(json_dict, json_file, indent=3)


z_sl = slice(mask_start[0], mask_start[0]+mask_size[0])
y_sl = slice(mask_start[1], mask_start[1]+  mask_size[1])
x_sl = slice(mask_start[2], mask_start[2]+mask_size[2])

for mode in ['seg','gt']:
    img = tiff.imread(path[mode])
    print(img.shape)
    img_crop = img[z_sl, y_sl, x_sl] if use_mask else img
    img_crop = label(img_crop).astype(np.uint16)
    print(img_crop.shape)

    if mode == 'seg':
        path_out_txt = os.path.join(path_out, '1_RES', 'res_track.txt')
        path_out_img = os.path.join(path_out, '1_RES', 'mask000.tif')
    elif mode == 'gt':
        path_out_txt = os.path.join(path_out, '1_GT', 'TRA', 'man_track.txt')
        path_out_img = os.path.join(path_out, '1_GT', 'TRA', 'man_track000.tif')
    
    
    
    props = regionprops(img_crop)
    txt_info = []
    for prop in props:
        txt_info.append('{} {} {} {}\n'.format(prop.label, 0, 0, 0))


    os.makedirs(os.path.dirname(path_out_txt))
    with open(path_out_txt, 'w') as f:
        f.writelines(txt_info)
    tiff.imwrite(path_out_img, img_crop)

if path['raw'] is not None:
    img = tiff.imread(path['raw'])
    img_crop = img[z_sl, y_sl, x_sl] if use_mask else img
    tiff.imwrite(os.path.join(path_out,'raw.tif'), img_crop)