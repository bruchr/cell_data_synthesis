import os

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.ndimage import binary_dilation, generate_binary_structure, binary_erosion

from image_data_class import Image_Data


'''
Compares the difference of distributions with the wasserstein distance.
Regions:
    - Background outside spheroid (manual def region)
    - Background inside spheroid (closed segmentation - initial segmentation)
    - Foreground inside spheroid (segmentation)
'''


class Img_Measure_WS_Dist():
    def __init__(self, img_data_real:Image_Data, img_data_syn:Image_Data) -> None:
        self.real_data = img_data_real
        self.syn_data = img_data_syn
        self.results = None
        self.calc_measure()

    def calc_measure(self):
        z_max = self.real_data.img_shape[0]
        self.results = {
            'bg_outer': np.zeros(z_max),
            'bg_inner': np.zeros(z_max),
            'fg_inner': np.zeros(z_max),
        }
        
        normalize = True
        adjust_mean = False
        adjust_mean_global = False

        img_real = self.real_data.get_image()
        img_syn = self.syn_data.get_image()
        seg_real = self.real_data.get_segmentation()
        seg_syn = self.syn_data.get_segmentation()
        seg_real_inner, seg_real_outer = Img_Measure_WS_Dist.seg_inner_outer(seg_real)
        seg_syn_inner, seg_syn_outer = Img_Measure_WS_Dist.seg_inner_outer(seg_syn)

        if adjust_mean_global:
            for z in range(img_real.shape[0]):
                img_real_mean_z = img_real[z,...].mean()
                img_syn[z,...] = img_syn[z,...] + (-img_syn[z,...].mean() + img_real_mean_z)

        # BG Outside
        for z in range(z_max):
            self.results['bg_outer'][z] =  Img_Measure_WS_Dist.get_w_dist_seg(img_real, img_syn, seg_real_outer, seg_syn_outer, z, normalize=normalize, adjust_mean=adjust_mean)

        # BG Inside
        for z in range(z_max):
            self.results['bg_inner'][z] = Img_Measure_WS_Dist.get_w_dist_seg(img_real, img_syn, seg_real_inner, seg_syn_inner, z, normalize=normalize, adjust_mean=adjust_mean)

        # FG Inner
        for z in range(z_max):
            self.results['fg_inner'][z] = Img_Measure_WS_Dist.get_w_dist_seg(img_real, img_syn, seg_real, seg_syn, z, normalize=normalize, adjust_mean=adjust_mean)


    @staticmethod
    def get_w_dist_seg(img_real, img_syn, seg_real, seg_syn, z, normalize=False, adjust_mean=False):
        wd = dict()
        real_seg_vals = Img_Measure_WS_Dist.get_seg_values(img_real, seg_real, z)
        syn_seg_vals = Img_Measure_WS_Dist.get_seg_values(img_syn, seg_syn, z)
        if adjust_mean:
            mean_real = real_seg_vals.mean()
            syn_seg_vals = syn_seg_vals + (-syn_seg_vals.mean() + mean_real)
        if real_seg_vals.size != 0 and syn_seg_vals.size != 0:
            wd_syn = wasserstein_distance(real_seg_vals, syn_seg_vals)
            if normalize:
                wd_black = wasserstein_distance(real_seg_vals, np.zeros(np.size(real_seg_vals), dtype=np.float64))
                wd_syn = 1 - wd_syn/wd_black
        else:
            wd_syn = None

        return wd_syn


    @staticmethod
    def get_seg_values(img, seg, z):
        img_ = img[z,...]
        seg_ = seg[z,...]
        pnts = img_[seg_]
        return pnts


    @staticmethod
    def seg_inner_outer(seg):
        srel = generate_binary_structure(3,1)
        seg_ = np.pad(seg, ((16,16), (0,0), (0,0)))
        seg_ = binary_dilation(seg_, srel, iterations=15)
        seg_ = binary_erosion(seg_, srel, iterations=15)
        seg_ = seg_[16:-16,...] # remove introduced padding

        seg_inner = seg_
        seg_outer = seg_==0
        seg_inner[seg==1] = 0
        seg_inner = binary_erosion(seg_inner, srel, iterations=2)
        seg_outer = binary_erosion(seg_outer, srel, iterations=2)

        return seg_inner, seg_outer



def plot_function_combined_box(image_data_list, image_measure_list, save_path=None):
    
    fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(5.5, 3)) # each type in a seperate plot
    bx_legend = {}
    for ind_typ, typ in enumerate(['bg_outer', 'bg_inner', 'fg_inner']):
        # get max z range of data for splitting
        max_len_res = 0
        for mode in ['sim', 'opti']:
            max_len_res_ = np.max([len(img_measure.results[typ]) for img_measure in image_measure_list[mode]])
            max_len_res = np.maximum(max_len_res, max_len_res_)
        # calculate 3 split points
        z_lims = np.multiply((1,2), int(np.round(max_len_res/3)))

        for mode in ['sim', 'opti']:
            res_reg = [[] for i in range(3)] # for each z region
            img_measure_list = image_measure_list[mode]
            
            for ind in range(len(img_measure_list)):
                results = img_measure_list[ind].results[typ]
                res_reg[0].append(results[:z_lims[0]])
                res_reg[1].append(results[z_lims[0]:z_lims[1]])
                res_reg[2].append(results[z_lims[1]:])

            res_reg[0] = np.concatenate(res_reg[0], axis=0)
            res_reg[1] = np.concatenate(res_reg[1], axis=0)
            res_reg[2] = np.concatenate(res_reg[2], axis=0)

            res_reg[0] = res_reg[0][~np.isnan(res_reg[0])]            
            res_reg[1] = res_reg[1][~np.isnan(res_reg[1])]
            res_reg[2] = res_reg[2][~np.isnan(res_reg[2])]
            # Remove nans from data
            # for res_r in res_reg:
            #     res_r = res_r[~np.isnan(res_r)]
            
            # Plot data on mode specific position
            get_pos = lambda x, mode: [x-0.125] if mode=='sim' else [x+0.125]
            bx0 = ax[ind_typ].boxplot(res_reg[0], positions=get_pos(0,mode), patch_artist=True, flierprops={'marker':'.'})
            bx1 = ax[ind_typ].boxplot(res_reg[1], positions=get_pos(0.75,mode), patch_artist=True, flierprops={'marker':'.'})
            bx2 = ax[ind_typ].boxplot(res_reg[2], positions=get_pos(1.5,mode), patch_artist=True, flierprops={'marker':'.'})

            # Set color of boxes based on the mode
            color = 'tab:orange' if mode=='sim' else 'tab:blue'
            for box in [bx0, bx1, bx2]:
                box['boxes'][0].set_facecolor(color)
                box['medians'][0].set_color('black')

            if ind_typ==0:
                bx_legend[mode] = bx0["boxes"][0]

        ax[ind_typ].set_xticks([0, 0.75, 1.5])
        ax[ind_typ].set_xlim([-0.30, 1.80])
        ax[ind_typ].set_xticklabels(['Up', 'Mid', 'Low'])
        if ind_typ == 0:
            ax[ind_typ].set_ylabel('Norm. Wasserstein Metric')
        ax[ind_typ].set_ylim((0.2, 1))
        ax[ind_typ].set_title(region_titles(typ))
        ax[ind_typ].grid(axis='y')

    ax[0].legend([bx_legend['sim'], bx_legend['opti']], ['Naive', 'Optimized'])
    
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(os.path.join(save_path, 'fig_combined_box.svg'))
        fig.savefig(os.path.join(save_path, 'fig_combined_box.pdf'))
        fig.savefig(os.path.join(save_path, 'fig_combined_box.png'))
    plt.show()

def region_titles(short):
    if short == 'bg_outer':
        return 'BG Outside'
    elif short == 'bg_inner':
        return 'BG Inside'
    elif short == 'fg_inner':
        return 'FG Inside'
    else:
        raise ValueError('Short form "{}" not known'.format(short))

if __name__ == '__main__':

    ########
    save_path = './Evaluation/Results/wasserstein_distance/'
    ########

    image_data_list = {
        'real': [],
        'sim': [],
        'opti': [],
    }
    # Real
    for im_nr in range(4):
        img_path = './Data/Real/{}_DAPI.tif'.format(im_nr+1)
        seg_path = './Data/Real/Segmentation/pp_{}_DAPI.tif'.format(im_nr+1)
        image_data_list['real'].append(Image_Data(img_path, seg_path, 'real'))

    # Simulation
    for im_nr in range(4):
        img_path = './Data/Simulated/images/00{}_001_final.tif'.format(im_nr)
        seg_path = './Data/Simulated/labels/00{}_001_final_label.tif'.format(im_nr)
        image_data_list['sim'].append(Image_Data(img_path, seg_path, 'sim'))

    # Optimization
    for im_nr in range(4):
        img_path = './Data/Optimized/transformed_00{}_001_final.tif'.format(im_nr)
        seg_path = './Data/Simulated/labels/00{}_001_final_label.tif'.format(im_nr)
        image_data_list['opti'].append(Image_Data(img_path, seg_path, 'opti'))


    image_measure_list = {
        'sim': [],
        'opti': [],
    }
    for mode in ['sim', 'opti']:
        for img_data_real, img_data_syn in zip(image_data_list['real'], image_data_list[mode]):
            image_measure_list[mode].append(Img_Measure_WS_Dist(img_data_real, img_data_syn))

    data_dict = {'image_data_list': image_data_list, 'image_measure_list': image_measure_list}
    os.makedirs(save_path, exist_ok=True)


    plot_function_combined_box(image_data_list, image_measure_list, save_path)