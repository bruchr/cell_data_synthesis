import numpy as np
from numpy.lib.npyio import save
import skimage.measure
import torch
from torch.nn.functional import conv3d
from scipy.ndimage import zoom

def convolve_label(label, psf, scale, t=0.25, sparse_labels=False, verbose=False):
    """
    Convolution of label images.
    Helpfull, if labels should match the size of nuclei in convolved nuclei image.
    Scale is necessary if image data was convolved in higher resolution than the output.
    
    Args:
        label: label image
        psf: psf image
        scale: needed if image convolution was performed in a different resolution than the output. Scale = px_size_desired / px_size_sim_img
        t: threshold used to convert convolution result back to a binary label
        sparse_labels: if label is of sparse type (label=1 is assumed to be background and label=0 to be ignored)
        verbose: if True, additional information is printed
    Returns:
        Image, with just the biggest object left
    """
    if scale != 0 or scale != 1:
        label = zoom(label, zoom=scale, order=0, prefilter=False)
    
    psf = psf/psf.sum()
    label_shape = label.shape
    if sparse_labels:
        label = label-1
    padding = tuple(np.floor(np.divide(psf.shape,2)).astype(int))
    psf = np.expand_dims(np.expand_dims(psf, axis=0), axis=0)
    label_new = np.zeros(np.shape(label))
    label_new_bin = np.zeros(np.shape(label), dtype=np.uint16)
    props = skimage.measure.regionprops(label)
    if verbose:
        print('Conv'+'-'*20)
        print('Conv: Input: Min/Max value in label: {} / {}. dtype: {}'.format(np.min(label), np.max(label), label.dtype))
        print('Conv: Shape of label: {}'.format(label.shape))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == 'cuda':
        torch.backends.cudnn.benchmark = True
    else:
        print('Conv: CPU is used. This may take a while.')

    for prop in props:
        # Convolution of each label on its own on a cropped image
        # label_loc = prop.image
        bb = np.asarray([prop.bbox[0:3], prop.bbox[3:6]])
        assert prop.image.shape == label[bb[0,0]:bb[1,0], bb[0,1]:bb[1,1], bb[0,2]:bb[1,2]].shape, 'Images are not the same'
        bb[0,:] = np.maximum(np.subtract(bb[0,:], padding), 0)
        bb[1,:] = np.minimum(np.add(bb[1,:], padding), label_shape)
        if verbose:
            print('Shape prop.region/ shape label[bbox]: {} / {}'.format(prop.image.shape, label[bb[0,0]:bb[1,0], bb[0,1]:bb[1,1], bb[0,2]:bb[1,2]].shape))
        label_loc = label[bb[0,0]:bb[1,0], bb[0,1]:bb[1,1], bb[0,2]:bb[1,2]] == prop.label
        assert np.count_nonzero(label_loc) == prop.area, 'Image shows more than one label {}/{}'.format(np.count_nonzero(label_loc), prop.area)
        label_loc = label_loc.astype(np.float32)
        label_loc = np.expand_dims(np.expand_dims(label_loc, axis=0), axis=0)
        
        label_loc_c = torch.from_numpy(label_loc).to(device)
        psf_c = torch.from_numpy(psf).to(device)
        label_loc = conv3d(label_loc_c, psf_c, padding=padding).cpu().numpy() # np.floor(psf.shape/2)
        label_loc = label_loc[0,0,:,:,:]


        label_loc_bin = (label_loc > t) * prop.label
        # label_loc_bin = (label_loc > 0.25) * prop.label

        c_l = np.asarray(np.where(label_loc_bin))
        # c_g = c_l+np.asarray(prop.bbox)[0:3, None]
        c_g = c_l+bb[0,0:3,None]


        label_new_bin[c_g[0,:],c_g[1,:],c_g[2,:]] = np.where(label_loc[c_l[0,:],c_l[1,:],c_l[2,:]] > label_new[c_g[0,:],c_g[1,:],c_g[2,:]],
            label_loc_bin[c_l[0,:],c_l[1,:],c_l[2,:]], label_new_bin[c_g[0,:],c_g[1,:],c_g[2,:]])
        label_new[c_g[0,:],c_g[1,:],c_g[2,:]] = np.where(label_new[c_g[0,:],c_g[1,:],c_g[2,:]] < label_loc[c_l[0,:],c_l[1,:],c_l[2,:]],
            label_loc[c_l[0,:],c_l[1,:],c_l[2,:]], label_new[c_g[0,:],c_g[1,:],c_g[2,:]])

    if sparse_labels:
        label_new_bin += 1
    label_new = zoom(label_new, zoom=np.divide(1,scale))
    label_new_bin = zoom(label_new_bin, zoom=np.divide(1,scale), order=0, prefilter=False)

    return label_new.astype(np.float32), label_new_bin



if __name__=='__main__':
    import os
    import tifffile as tiff

    path = 'path/to/labels/folder'
    sparse_labels = True
    scale = (2.05355963896, 2.86436498217, 2.86436498217)
    # scale: needed if image convolution was performed in a different resolution than the output. Scale = px_size_desired / px_size_sim_img
    save_path = 'path/to/convolved/labels/folder'
    path_psf = 'path/to/psf'

    os.makedirs(save_path, exist_ok=True)

    for f_name in os.listdir(path):
        if f_name.endswith('.tif'):
            print(f_name)
            f_path = os.path.join(path, f_name)
            
            label = tiff.imread(f_path)

            _, label_bin = convolve_label(label, tiff.imread(path_psf), sparse_labels=sparse_labels, scale=scale, t=0.4)

            tiff.imwrite(os.path.join(save_path, f_name.replace('.tif','_conv.tif')), label_bin)