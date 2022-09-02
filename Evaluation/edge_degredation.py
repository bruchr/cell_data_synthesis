import os

from matplotlib import pyplot as plt
import numpy as np
from skimage import filters

from image_data_class import Image_Data
from sliding_window import generateSlices


class Img_Measure_Edge_Qual():
    def __init__(self, img_data:Image_Data) -> None:
        self.img_data = img_data
        self.results = None
        self.calc_measure()

    def calc_measure(self):
        img = self.img_data.get_image()

        crop_size = (img.shape[0],100,100)
        crop_rest = np.remainder(img.shape[1:], crop_size[1:])/2
        sl = [
            slice(int(np.floor(crop_rest[0])),  int(-np.ceil(crop_rest[0]))),
            slice(int(np.floor(crop_rest[0])),  int(-np.ceil(crop_rest[0]))),
        ]
        img = img[:, sl[0], sl[1]]

        img_gauss = filters.gaussian(img, (0,3,3))
        img_edge = np.zeros_like(img)
        for z_ind in range(img.shape[0]):
            img_edge[z_ind]
            img_edge[z_ind, ...] = filters.sobel(img_gauss[z_ind, ...])
        
        slices, _ = generateSlices(img.shape, crop_size, 0)

        results = []
        for sl in slices:
            img_edge_sl = img_edge[sl[0], sl[1], sl[2]]
            # seg_sl = seg[sl[0], sl[1], sl[2]]
            res = self.__calc_measure_region(img_edge_sl)
            if res is not None:
                results.append(res)
        
        self.results = self.__calc_area_mean(results)


    def __calc_measure_region(self, img_edge):
        results = np.quantile(img_edge, 0.95, axis=(1,2))
        max_ind = np.argmax(results)
        results = results[max_ind:]
        if results.size < 50 or results.max() < 0.026:
            results = None
        return results
        

    def __calc_area_mean(self, results):
        res_max_len = np.max([len(res) for res in results])
        n_vals = np.zeros(res_max_len)
        result = np.zeros(res_max_len)

        for res in results:
            n_vals[:len(res)] += 1
            result[:len(res)] += res
        
        result /= n_vals
        return result

def split_list(image_data_list, image_measure_list, types=('real', 'sim', 'opti')):
    out_dict_data = {}
    out_dict_measure = {}
    for typ in types:
        data_list = []
        measure_list = []
        for ind, image_data in enumerate(image_data_list):
            if image_data.img_type == typ:
                data_list.append(image_data)
                measure_list.append(image_measure_list[ind])
        out_dict_data[typ] = data_list
        out_dict_measure[typ] = measure_list

    return out_dict_data, out_dict_measure

def calc_mean_uneven(results):
    res_max_len = np.max([len(res) for res in results])
    n_vals = np.zeros(res_max_len)
    result = np.zeros(res_max_len)
    for res in results:
        n_vals[:len(res)] += 1
        result[:len(res)] += res
    result /= n_vals
    return result

def plot_function_normal_mean(image_data_list, image_measure_list, save_path=None):
    fig, ax = plt.subplots(figsize=(5.5,  3))
    # comb_res = np.zeros((len(image_data_list), 3))
    max_len_res = np.max([len(img_measure.results) for img_measure in image_measure_list])

    data_dict, measure_dict = split_list(image_data_list, image_measure_list)

    real_measure_list = []
    sim_measure_list = []
    opti_measure_list = []

    for ind in range(len(data_dict['real'])):

        real_measure_list.append(measure_dict['real'][ind])
        sim_measure_list.append(measure_dict['sim'][ind])
        opti_measure_list.append(measure_dict['opti'][ind])
    
    counter = 0
    for measure_list, col in zip((real_measure_list, sim_measure_list, opti_measure_list), ('black','tab:orange','tab:blue')):
        m_len = np.max([len(measure.results) for measure in measure_list])
        measure_array = np.empty((m_len, len(measure_list)))
        measure_array[:] = np.NaN
        for ind_m, measure in enumerate(measure_list):
            measure_array[:len(measure.results), ind_m] = measure.results

        measure_mean = np.nanmean(measure_array, axis=1)
        measure_std = np.nanstd(measure_array, axis=1)
        
        ax.errorbar(np.arange(len(measure_mean))+counter, measure_mean, yerr=measure_std, color=col)
        counter += 0.1

    ax.set_xlabel('Normalized Z-Index')
    ax.set_ylabel('q95 Edge Quality')
    ax.legend(['Real', 'Naive', 'Optimized'])
    
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(os.path.join(save_path, 'fig_normal_mean.svg'))
        fig.savefig(os.path.join(save_path, 'fig_normal_mean.pdf'))
        fig.savefig(os.path.join(save_path, 'fig_normal_mean.png'))

    plt.show()

def plot_function_combined_box(image_data_list, image_measure_list, save_path=None):
    fig, ax = plt.subplots(figsize=(5.5,  3))
    # comb_res = np.zeros((len(image_data_list), 3))
    max_len_res = np.max([len(img_measure.results) for img_measure in image_measure_list])

    z_lims = np.multiply((1,2), int(np.round(max_len_res/3)))
    print('Z Limits: {}, Max Z-Ind: {}'.format(z_lims, max_len_res))

    data_dict, measure_dict = split_list(image_data_list, image_measure_list)

    for typ in ['sim', 'opti']:
        res_reg = [[] for i in range(3)]
        for ind in range(len(data_dict['real'])):
            real_data = data_dict['real'][ind]
            real_measure = measure_dict['real'][ind]
            other_data = data_dict[typ][ind]
            other_measure = measure_dict[typ][ind]

            mlen = np.minimum(len(other_measure.results), len(real_measure.results))
            results = np.abs(other_measure.results[:mlen] - real_measure.results[:mlen])

            res_reg[0].append(results[:z_lims[0]])
            res_reg[1].append(results[z_lims[0]:z_lims[1]])
            res_reg[2].append(results[z_lims[1]:])

        res_reg[0] = np.concatenate(res_reg[0], axis=0)
        res_reg[1] = np.concatenate(res_reg[1], axis=0)
        res_reg[2] = np.concatenate(res_reg[2], axis=0)
        

        get_pos = lambda x, typ: [x] if typ=='sim' else [x+0.25]
        get_label = lambda typ: 'Simulated' if typ =='sim' else 'Optimized'
        bx0 = ax.boxplot(res_reg[0], positions=get_pos(0,typ), patch_artist=True)
        bx1 = ax.boxplot(res_reg[1], positions=get_pos(1,typ), patch_artist=True)
        bx2 = ax.boxplot(res_reg[2], positions=get_pos(2,typ), patch_artist=True)


        color = 'tab:orange' if typ=='sim' else 'tab:blue'
        for box in [bx0, bx1, bx2]:
            box['boxes'][0].set_facecolor(color)
            box['medians'][0].set_color('black')

        if typ=='sim':
            bx_sim = bx0["boxes"][0]
        elif typ=='opti':
            bx_opti = bx0["boxes"][0]
        
    ax.legend([bx_sim, bx_opti], ['Naive', 'Optimized'])
    ax.grid(axis='y')

    ax.set_xticks([0.125, 1.125, 2.125])
    ax.set_xticklabels(['Upper', 'Middle', 'Lower'])
    ax.set_ylim((0, 0.016))
    ax.set_ylabel('Difference q95 Edge Quality')

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(os.path.join(save_path, 'fig_combined_box.svg'))
        fig.savefig(os.path.join(save_path, 'fig_combined_box.pdf'))
        fig.savefig(os.path.join(save_path, 'fig_combined_box.png'))

    plt.show()


if __name__ == '__main__':

    ########
    save_path = './Evaluation/Results/edge_degredation/'
    ########

    image_data_list = []
    for im_nr in range(4):
        img_path = './Data/Real/{}_DAPI.tif'.format(im_nr+1)
        seg_path = './Data/Real/Segmentation/pp_{}_DAPI.tif'.format(im_nr+1)
        image_data_list.append(Image_Data(img_path, seg_path, 'real'))

    for im_nr in range(4):
        img_path = './Data/Simulated/images/00{}_001_final.tif'.format(im_nr)
        seg_path = './Data/Simulated/labels/00{}_001_final_label.tif'.format(im_nr)
        image_data_list.append(Image_Data(img_path, seg_path, 'sim'))

    for im_nr in range(4):
        img_path = './Data/Optimized/transformed_00{}_001_final.tif'.format(im_nr)
        seg_path = './Data/Simulated/labels/00{}_001_final_label.tif'.format(im_nr)
        image_data_list.append(Image_Data(img_path, seg_path, 'opti'))


    image_measure_list = []
    for image_data in image_data_list:
        image_measure_list.append(Img_Measure_Edge_Qual(image_data))

    os.makedirs(save_path, exist_ok=True)


    plot_function_normal_mean(image_data_list, image_measure_list, save_path)
    plot_function_combined_box(image_data_list, image_measure_list, save_path)