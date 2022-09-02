import numpy as np
import tifffile as tiff


class Image_Data():
    def __init__(self, img_path:str, seg_path:str, img_type:str) -> None:
        self.img_path = img_path
        self.seg_path = seg_path
        self.img_type = img_type
        self.min_shape = None

        tmp = self.get_image()
        self.img_shape = tmp.shape
        tmp = self.get_segmentation()
        seg_shape = tmp.shape
        if not np.array_equal(self.img_shape, seg_shape):
            print('#### Warning ####\nImage and segmentation need to be of the same shape, but are {} and {}'.format(self.img_shape, seg_shape))
            self.min_shape = np.minimum(self.img_shape, seg_shape)
            self.img_shape = self.min_shape
            # raise Warning('Image and segmentation need to be of the same shape')

        self.def_plot_info()
        

    def get_image(self):
        if self.img_type=='opti':
            img = tiff.imread(self.img_path).astype(np.float32)/65535
        else:
            img = tiff.imread(self.img_path).astype(np.float32)/255
        if self.min_shape is not None:
            img = img[:self.min_shape[0], :self.min_shape[1], :self.min_shape[2]]
        return img

    def get_segmentation(self):
        img = tiff.imread(self.seg_path)!=0
        if self.min_shape is not None:
            img = img[:self.min_shape[0], :self.min_shape[1], :self.min_shape[2]]
        return img

    def def_plot_info(self) -> None:
        if self.img_type=='real':
            self.color='black'
        elif self.img_type=='sim':
            self.color='tab:orange'
        elif self.img_type=='opti':
            self.color='tab:blue'
        else:
            self.color='black'