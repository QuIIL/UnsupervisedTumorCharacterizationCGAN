import argparse
import glob
import os
import re
import warnings

import cv2
import numpy as np
import torch
import torch.utils.data as data
from imgaug import augmenters as iaa
from progress.bar import Bar as ProgressBar

from dataset import SerialLoader
from config import Config
from misc import utils
from model import netdesc
from stats import get_patch_stat

warnings.filterwarnings('ignore',category=UserWarning)

class Inferer(Config):
    def __infer_step(self, input_batch, net):
        net.eval() # set to eval mode

        msks_imgs = input_batch

        msks_imgs = torch.tensor(msks_imgs)
        msks_imgs = msks_imgs.permute(0, 3, 1, 2) # to NCHW
        # push data to GPUs and convert to float32
        msks_imgs = msks_imgs.to('cuda').float() / 255.0
        
        with torch.no_grad(): # dont compute gradient
            fake_imgs = net(msks_imgs)
        fake_imgs = fake_imgs.permute(0, 2, 3, 1)
        fake_imgs = fake_imgs.cpu().numpy()
        fake_imgs[fake_imgs < 0] = 0.0    
        fake_imgs[fake_imgs > 1] = 1.0    
        fake_imgs = (fake_imgs * 255.0).astype('uint8')
        return fake_imgs
    ####

    def __run_list(self, net, dataloader, output_dir):
        pbar = ProgressBar('Processing', max=len(dataloader), width=48)
        for _, batch_data in enumerate(dataloader):
            imgs_input, codenames, sizes = batch_data  
            imgs_recon = self.__infer_step(imgs_input, net)          
            for idx in range(0, imgs_recon.shape[0]):
                orig_img_size = sizes[idx].numpy()
                recon_img = cv2.cvtColor(imgs_recon[idx], cv2.COLOR_RGB2BGR)
                recon_img = utils.cropping_center(recon_img, orig_img_size)
                cv2.imwrite('%s/%s.jpg' % (output_dir, codenames[idx]), recon_img)
            pbar.next()
        pbar.finish()
        return        
    ####
    def __infer(self, inf_model_path, input_file_list, pred_out_dir):      

        augmentors = self.infer_augmentors()
        saved_state_dict = torch.load(inf_model_path)
        net_g = netdesc.Generator(1, 3).to('cuda')
        net_g.load_state_dict(saved_state_dict['net_g'])
        net_g = torch.nn.DataParallel(net_g).to('cuda')

        filepath_list = input_file_list
        basename_list = [os.path.basename(path).split('.')[0] for path in input_file_list]
        input_info = list(zip(filepath_list, basename_list))

        infer_dataset = SerialLoader(
                input_info, run_mode='infer',
                shape_augs=iaa.Sequential(augmentors[0]),
                input_augs=iaa.Sequential(augmentors[1]))

        dataloader = data.DataLoader(infer_dataset, 
                        num_workers=self.nr_procs_infer, 
                        batch_size=1, 
                        drop_last=False)

        utils.rm_n_mkdir(pred_out_dir)
        self.__run_list(net_g, dataloader, pred_out_dir)

        return

    def run_infer_cv(self):
        # axis 0 is RGB(0) or GRAY(1)
        # axis 1 is CC(0) or MI(1) or SSIM(2)
        metric_idx = [0, 0] # select CC of RGB

        # loop over N-checkpoint of each k-fold to perform evaluation
        # on the inner validation set
        for fold_idx in range(0, self.kfold['nr_fold']):
            for fold_idy in range(0, self.kfold['nr_fold']):

                src_file_list = self.dataset.get_subset(
                                    nr_fold=self.kfold['nr_fold'],
                                    fold_idx=[fold_idx, fold_idy], 
                                    mode='valid')

                # * pre-caching the file path
                rec_file_list = [os.path.basename(path) for path in src_file_list]
                rec_file_list = [path.split('.')[0] for path in rec_file_list]
                rec_file_list = ['%s/%s.jpg' % (self.temp_rec_img_dir, path) for path in rec_file_list]

                log_dir = '%s/%02d_%02d/' % (self.log_dir, fold_idx, fold_idy)
                chkpt_path_list = glob.glob(log_dir + '/*.pth')
                chkpt_path_list.sort()

                stat_path_list = []
                for chkpt_path in chkpt_path_list:
                    print(chkpt_path)
                    chkpt_id = os.path.basename(chkpt_path)
                    chkpt_id = chkpt_id.split('.')[0].split('_')[-1]

                    self.__infer(chkpt_path, src_file_list, self.temp_rec_img_dir)

                    # ! training was done on benign (label 0) only, so hard code return `label_id`
                    fold_stat_save_path = '%s/%02d_%02d_%s' % (self.cv_stat_valid_dir, fold_idx, fold_idy, chkpt_id)                                                                
                    get_patch_stat.run(src_file_list, rec_file_list, fold_stat_save_path, encode_label=0)
                    stat_path_list.append(fold_stat_save_path + '.npy')

                # * load the `valid` output file of epoch above and
                # * find the best epoch to be used for `test` or
                # * evaluation on other dataset

                run_stat_list = [np.load(path, allow_pickle=True) for path in stat_path_list] 
                run_stat_list = np.array(run_stat_list)[...,:2]
                run_stat_list = np.array(run_stat_list.tolist())
                run_stat_list = run_stat_list[:,:,metric_idx[0], metric_idx[1]]
                # select CV checkpoint path
                chkpt_idx = np.argmax(np.mean(run_stat_list, axis=1))

                # * now run evaluation on the withold test set of nested cv
                src_file_list = self.dataset.get_subset(
                                    nr_fold=self.kfold['nr_fold'],
                                    fold_idx=[fold_idx, fold_idy], 
                                    mode='test')

                rec_file_list = [os.path.basename(path) for path in src_file_list]
                rec_file_list = [path.split('.')[0] for path in rec_file_list]
                rec_file_list = ['%s/%s.jpg' % (self.temp_rec_img_dir, path) for path in rec_file_list]

                chkpt_path = chkpt_path_list[chkpt_idx]
                self.__infer(chkpt_path, src_file_list, self.temp_rec_img_dir)

                # ! training was done on benign (label 0) only, so hard code return `label_id`
                fold_stat_save_path = '%s/%02d_%02d_%s' % (self.cv_stat_test_dir, fold_idx, fold_idy, chkpt_id)                                                                            
                get_patch_stat.run(src_file_list, rec_file_list, fold_stat_save_path, encode_label=0)
        return

    def run_infer(self):
        """
        A more generic inference version
        """

        for dir_info in self.infer_input_dir_info_list:
            print(dir_info[0])
            src_file_list = glob.glob('%s/*.tif' % dir_info[0])
            src_file_list.sort()
            
            dir_name = re.split(r'[/]', dir_info[0])[-2]
            rec_img_dir = '%s/%s/' % (self.infer_output_root_dir, dir_name)

            rec_file_list = [os.path.basename(path) for path in src_file_list]
            rec_file_list = [path.split('.')[0] for path in rec_file_list]
            rec_file_list = ['%s/%s.jpg' % (rec_img_dir, path) for path in rec_file_list]

            self.__infer(self.infer_chkpt_path, src_file_list, rec_img_dir)
            stat_save_path = '%s/%s.npy' % (self.infer_output_root_dir, dir_name)                                                                           
            get_patch_stat.run(src_file_list, rec_file_list, stat_save_path, encode_label=dir_info[1])
        return

####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--mode', help='`generic` or `self-cv`, `generic` for running on any dataset'
                                    'while `self-cv` for getting nested-cv evaluation after training.')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    inferer = Inferer()
    if args.mode == 'self-cv':
        inferer.run_infer_cv()   
    else:
        inferer.run_infer()
