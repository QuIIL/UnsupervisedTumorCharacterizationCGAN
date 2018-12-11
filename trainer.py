
import copy
import argparse
import importlib
import json
import os
import warnings

import torch.utils.data as data
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
# https://github.com/aleju/imgaug
from imgaug import augmenters as iaa
from tensorboardX import SummaryWriter

from dataset import SerialLoader
from config import Config
from misc.train_utils import (ScalarMovingSummary, check_manual_seed,
                              log_train_ema_results)
from misc.utils import rm_n_mkdir
from model import netdesc

####
class Trainer(Config):
    ####
    def get_datagen(self, batch_size, view=False, **kwargs):
        
        assert kwargs['mode'] == 'train'
        train_paths = self.dataset.get_subset(**kwargs)

        augmentors = self.train_augmentors()
        train_dataset = SerialLoader(train_paths[:], 
                        shape_augs=iaa.Sequential(augmentors[0]),
                        input_augs=iaa.Sequential(augmentors[1]))
        if view:
            return train_dataset

        nr_procs = self.nr_procs_train
        train_loader = data.DataLoader(train_dataset, 
                                    num_workers=nr_procs, 
                                    batch_size=batch_size, 
                        shuffle=True, drop_last=True)
        return train_loader

    ####
    def run_once(self, opt, log_dir, dataset_opt):
        
        check_manual_seed(self.seed)

        # --------------------------- Dataloader
        train_loader = self.get_datagen(opt['train_batch_size'], **dataset_opt)

        # --------------------------- Training Sequence

        if self.logging:
            rm_n_mkdir(log_dir)

        ####
        net_g = netdesc.Generator(1, 3).to('cuda')
        net_d = netdesc.Discriminator(4).to('cuda')

        # TODO: more flexible scheduler
        optimizer_g, optimizer_args = opt['optimizer_g']
        optimizer_g = optimizer_g(net_g.parameters(), **optimizer_args)
        scheduler_g = opt['scheduler_g'](optimizer_g)

        optimizer_d, optimizer_args = opt['optimizer_d']
        optimizer_d = optimizer_d(net_d.parameters(), **optimizer_args)
        scheduler_d = opt['scheduler_d'](optimizer_d)

        # NOTE: unpacking need to match with what defined in train step, using dict to ensure?
        run_step = importlib.import_module('model.run_step')

        # ! tuple is immutable so the nets wont be updated !!!
        train_info = {'g_info' : [net_g, optimizer_g], 
                      'd_info' : [net_d, optimizer_d],
                      'extra_train_opt' : opt['extra_train_opt']}
        # train_info = [net_g, net_d, optimizer_g, optimizer_d]
        train_step = run_step.__getattribute__('train_step')
        train_engine = Engine(lambda engine, batch: train_step(engine, batch, train_info))
        ####

        ####
        pbar = ProgressBar(persist=True)
        pbar.attach(train_engine)

        timer = Timer(average=True)
        timer.attach(train_engine, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                            pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

        if self.logging:
            checkpoint_handler = ModelCheckpoint(log_dir, self.chkpts_prefix,                                     
                                    save_as_state_dict=True,
                                    save_interval=1, n_saved=50, 
                                    global_step_transform=lambda engine, event: engine.state.epoch,
                                    require_empty=False)
            # adding handlers using `trainer.add_event_handler` method API
            train_engine.add_event_handler(
                                    event_name=Events.EPOCH_COMPLETED(every=1), 
                                    handler=checkpoint_handler,
                                    to_save={'net_g': net_g, 
                                             'net_d': net_d,
                                             }) 

            tfwriter = SummaryWriter(log_dir=log_dir)
            json_log_file = log_dir + '/stats.json'
            with open(json_log_file, 'w') as json_file:
                json.dump({}, json_file) # create empty file

        # adding handlers using `trainer.on` decorator API
        @train_engine.on(Events.EXCEPTION_RAISED)
        def handle_exception(engine, e):
            if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
                warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')
            else:
                raise e

        log_info_dict = {
            'logging'      : self.logging,
            'optimizer_g'  : optimizer_g,
            'optimizer_d'  : optimizer_d,
            'tfwriter'     : None if not self.logging else tfwriter,
            'json_file'    : None if not self.logging else json_log_file,
        }

        ema_calculator = ScalarMovingSummary(alpha=0.95)
        train_engine.add_event_handler(Events.ITERATION_COMPLETED, lambda e : ema_calculator.update(e))

        # to change the lr
        train_engine.add_event_handler(Events.EPOCH_STARTED, 
                        lambda e : scheduler_g.step() if e.state.epoch < 100 else None)
        train_engine.add_event_handler(Events.EPOCH_STARTED, 
                        lambda e : scheduler_d.step() if e.state.epoch < 100 else None)

        # to monitor the lr
        def track_variables(engine, name, state):
            engine.state.metrics[name] = state['lr'] # HACK
        train_engine.add_event_handler(Events.EPOCH_COMPLETED, track_variables, 'lr_g', optimizer_g.param_groups[0]) 
        train_engine.add_event_handler(Events.EPOCH_COMPLETED, track_variables, 'lr_d', optimizer_g.param_groups[0])
        train_engine.add_event_handler(Events.EPOCH_COMPLETED, log_train_ema_results, log_info_dict)

        # Setup is done. Now let's run the training
        train_engine.run(train_loader, opt['nr_epochs'])
        return
    ####
    def run(self):

        phase_opts = self.training_phase
        assert len(phase_opts) == 1, 'Not implemented for more phases'

        if self.kfold is None:
            opt = phase_opts[0]
            self.run_once(opt, self.log_dir, opt['pretrained'])
        else: # do nested cv
            for fold_idx in range(0, self.kfold['nr_fold']):
                for fold_idy in range(0, self.kfold['nr_fold']):
                    opt = phase_opts[0]
                    dataset_opt = copy.deepcopy(self.kfold)
                    dataset_opt['mode'] = 'train'
                    dataset_opt['fold_idx'] = [fold_idx, fold_idy]
                    log_dir = '%s/%02d_%02d/' % (self.log_dir, fold_idx, fold_idy)
                    self.run_once(opt, log_dir, dataset_opt)
        return
    ####

####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    trainer = Trainer()
    trainer.run()
