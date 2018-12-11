
import importlib

from imgaug import augmenters as iaa
from dataset import ColonDataset

#### 
class Config(object):
    def __init__(self):
        self.seed    = 5
      
        # nr of processes for parallel processing input
        self.nr_procs_train = 8 
        self.nr_procs_infer = 0 

        #### Dynamically setting the config file into variable
        config_file = importlib.import_module('model.opt')
        config_dict = config_file.__getattribute__('config')
        for variable, value in config_dict.items():
            self.__setattr__(variable, value)

        self.dataset = ColonDataset()
        ####
        
        self.kfold = {
            'nr_fold' : 5,
        }
        self.logging = True
        self.log_path = 'exp_output/'
        self.chkpts_prefix = ''

        self.exp_name = 'lambda=120/'
        self.log_dir =  self.log_path + self.exp_name

        # * for inference on `valid` and `test` of nested cv
        self.temp_rec_img_dir  = 'exp_output/eval_cv_output/rec/'
        self.cv_stat_valid_dir = 'exp_output/eval_cv_output/valid_stat/'
        self.cv_stat_test_dir  = 'exp_output/eval_cv_output/test_stat/'

        # * for generic inference
        self.infer_output_root_dir = 'exp_output/eval_test_output/'
        self.infer_chkpt_path = 'exp_output/lambda=120/00_00/checkpoint_3.pth'
        # format [[dir_path, label_code]]
        # label_code = None to auto decode label of each image from its file name,
        # assuming in the form `*_X.jpg` with X is the label code e.g 0-N
        self.infer_input_dir_info_list = [        
            # ['', None],
            ['../../dataset/NUC_HE_Kumar/train-set/orig_split/train/', 0]
        ]        
        return

    def train_augmentors(self):
        shape_augs = [
            iaa.Affine(
                # scale images to X-Y% of their size, individually per axis
                rotate=(-179, 179), # rotate by -179 to +179 degrees
                order=[1],    # [0] is nearest, [1] is linear etc.
                backend='cv2' # opencv for fast processing
            ),
            iaa.size.CropToFixedSize(1024, 1024,
                                position='center', 
                                deterministic=True),
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.5), # vertically flip 20% of all images
        ]
        #
        input_augs = [
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        ]   
        return shape_augs, input_augs

    def infer_augmentors(self):
        shape_augs = None
        input_augs = None
        return shape_augs, input_augs
