
import json
import os
import random
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from termcolor import colored

####
class ScalarMovingSummary(object):
    def __init__(self, alpha=0.95):
        self.alpha = alpha
        self.tracking_dict = {}
    def update(self, engine):
        step_output_dict = engine.state.output['scalar']

        for key, current_value in step_output_dict.items():
            if key in self.tracking_dict:
                old_ema_value = self.tracking_dict[key]
                new_ema_value = old_ema_value * self.alpha + (1.0 - self.alpha) * current_value
                self.tracking_dict[key] = new_ema_value
            else: # init for variable appearing first time
                # TODO: double checking this
                new_ema_value = current_value 
                self.tracking_dict[key] = new_ema_value

        # so that only update tensorboard for giving values only
        # in case of not running same training loop such as GAN 
        engine.step_ema = self.tracking_dict
        return
####
def check_manual_seed(seed):
    """ 
    If manual seed is not specified, choose a random one and communicate it to the user.
    """

    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print('Using manual seed: {seed}'.format(seed=seed))
    return

####
def check_log_dir(log_dir):
    # check if log dir exist
    if os.path.isdir(log_dir):
        color_word = colored('WARNING', color='red', attrs=['bold', 'blink'])
        print('%s: %s exist!' % (color_word, colored(log_dir, attrs=['underline'])))
        while (True):
            print('Select Action: d (delete) / q (quit)', end='')
            key = input()
            if key == 'd':
                shutil.rmtree(log_dir)
                break
            elif key == 'q':
                exit()
            else:
                color_word = colored('ERR', color='red')
                print('---[%s] Unrecognize Characters!' % color_word)
    return

####
def update_log(output, epoch, prefix, color, tfwriter, log_file, logging):
    # print values and convert
    max_length = len(max(output.keys(), key=len))
    for metric in output:
        key = colored(prefix + '-' + metric.ljust(max_length), color)
        print('------%s : ' % key, end='')
        if metric != 'conf_mat':
            print('%0.7f' % output[metric])
        else:
            conf_mat = output['conf_mat'] # use pivot to turn back
            conf_mat_df = pd.DataFrame(conf_mat)
            conf_mat_df.index.name = 'True'
            conf_mat_df.columns.name = 'Pred'
            output['conf_mat'] = conf_mat_df
            print('\n', conf_mat_df)

    if not logging:
        return

    # create stat dicts
    stat_dict = {}
    for metric in output:
        if metric != 'conf_mat':
            metric_value = output[metric] 
        else:
            conf_mat_df = output['conf_mat'] # use pivot to turn back
            conf_mat_df = conf_mat_df.unstack().rename('value').reset_index()
            conf_mat_df = pd.Series({'conf_mat' : conf_mat}).to_json(orient='records')
            metric_value = conf_mat_df
        stat_dict['%s-%s' % (prefix, metric)] = metric_value

    # json stat log file, update and overwrite
    with open(log_file) as json_file:
        json_data = json.load(json_file)

    current_epoch = str(epoch)
    if current_epoch in json_data:
        old_stat_dict = json_data[current_epoch]
        stat_dict.update(old_stat_dict)
    current_epoch_dict = {current_epoch : stat_dict}
    json_data.update(current_epoch_dict)

    with open(log_file, 'w') as json_file:
        json.dump(json_data, json_file)

    # log values to tensorboard
    for metric in output:
        if metric != 'conf_mat':
            tfwriter.add_scalar(prefix + '-' + metric, output[metric], current_epoch)

def log_train_ema_results(engine, info):
    """
    running training measurement
    """
    update_log(engine.step_ema, engine.state.epoch, 'train-ema', 'green',
                info['tfwriter'], info['json_file'], info['logging'])

    if 'images' in engine.state.output:
        tracked_images = engine.state.output['images'] # NCHW
        tracked_images = np.concatenate([tracked_images[0], 
                                         tracked_images[1]], axis=1) # CHW
        info['tfwriter'].add_image('train/Image', tracked_images, engine.state.epoch)

####
def accumulate_output(engine):
    batch_output = engine.state.output
    for key, item in batch_output.items():
        item = item.tolist()
        if key not in engine.accumulator:              
            engine.accumulator[key] = item
        else:
            engine.accumulator[key].extend(item)
    return
####
def process_accumulated_output(output, nr_classes):

    prob = np.array(output['prob'])
    true = np.array(output['true'])
    # threshold then get accuracy
    pred = np.argmax(prob, axis=-1)
    acc = np.mean(pred == true)
    # confusion matrix
    conf_mat = confusion_matrix(true, pred, 
                        labels=np.arange(nr_classes))
    #
    proc_output = dict(acc=acc, conf_mat=conf_mat)
    return proc_output

####
# * datatype keyword and accompanied output function?
def inference(train_engine, infer_engine, dataloader, info):
    """
    inference measurement
    """
    # * init placeholder
    infer_engine.accumulator = {} # init
    infer_engine.run(dataloader)

    output_stat = process_accumulated_output(infer_engine.accumulator, info['nr_classes'])
    update_log(output_stat, train_engine.state.epoch, 'valid', 'red', 
                info['tfwriter'], info['json_file'], info['logging'])
    return
