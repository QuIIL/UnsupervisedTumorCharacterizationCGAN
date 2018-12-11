import torch.optim as optim

config = {
    'nr_classes' : 2,
    
    'training_phase' : [
        {
            'nr_epochs' : 150, 
            'optimizer_g'  : [
                optim.Adam,
                { # should match keyword for parameters within the optimizer
                    'lr'    : 1.0e-4, # initial learning rate,
                    'betas' : (0.5, 0.999)
                }
            ],
            'scheduler_g'  : lambda x : optim.lr_scheduler.StepLR(x, 50), # learning rate scheduler

            'optimizer_d'  : [
                optim.Adam,
                { # should match keyword for parameters within the optimizer
                    'lr'    : 1.0e-4, # initial learning rate,
                    'betas' : (0.5, 0.999)
                }
            ],
            'scheduler_d'  : lambda x : optim.lr_scheduler.StepLR(x, 50), # learning rate scheduler

            'extra_train_opt' : {
                'generator_period' : 1,
                'lambda' : 120
            },

            'train_batch_size' : 2,
            'infer_batch_size' : 1,

            # path to load, -1 to auto load checkpoint from previous phase, 
            # None to start from scratch
            'pretrained' : None,
        },

    ],
}
