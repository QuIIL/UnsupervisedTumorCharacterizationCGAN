
import sys

import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(formatter={'float': '{:0.4f}'.format})
np.set_printoptions(threshold=sys.maxsize)

####
def make_boxplot(data_dict, metric_titles):
    # expected, may not be aligned with data_dict
    nr_metric = len(metric_titles)

    color_gen = plt.cm.get_cmap('jet', len(list(data_dict.keys())))
    dataset_facecolor = {dataset_name : color_gen(idx) 
                            for idx, dataset_name in enumerate(list(data_dict.keys()))}

    placement = 1
    for img_type in range(0, 2):
        for score_type in range(0, nr_metric):
            ax = plt.subplot(2, nr_metric, placement)

            series_name_list = []
            variables_series = []
            series_facecolor_list = []
            for dataset_name, dataset in data_dict.items():
                label_code, label_name = zip(*dataset['label'].items())

                series_name_list.extend(label_name)
                sample_stat, sample_info, _ = dataset['info']

                facecolor = dataset['color'] if 'color' in dataset \
                                else dataset_facecolor[dataset_name]
                facecolor_list = [facecolor for i in range(0, len(label_name))]
                series_facecolor_list.extend(facecolor_list)

                for class_idx in label_code:
                    class_stat_idx = (sample_info[:,-1] == class_idx)
                    class_stat_idx = np.nonzero(class_stat_idx)[0]
                    class_stat = sample_stat[class_stat_idx, img_type]

                    # score_sample_list = np.abs(class_stat[:,score_type])
                    score_sample_list = class_stat[:,score_type]
                    if dataset['label'][class_idx][1] == 'BN' and dataset_name != 'train':
                        print(np.mean(score_sample_list), np.std(score_sample_list))
                    variables_series.append(score_sample_list)

            # print(len(series_name_list), len(variables_series))
            boxplot = plt.boxplot(
                variables_series, labels=series_name_list, 
                patch_artist=True, vert=True,
                flierprops=dict(markerfacecolor='r', marker='D', markeredgewidth=2),
            )

            for series_idx, _ in enumerate(series_facecolor_list):
                facecolor = series_facecolor_list[series_idx]
                boxplot['boxes'][series_idx].set_facecolor(facecolor)
                boxplot['medians'][series_idx].set_color('black')    

            ax.set_ylim(-0.05, 1.05)
            plt.tick_params(axis='both', which='major')
            plt.tick_params(axis='both', which='minor')
            plt.title(metric_titles[score_type])
            placement += 1
####

train_set_output_file_path = None
eval_set_output_file_path = None
train_info = np.load(train_set_output_file_path)
eval_info = np.load(eval_set_output_file_path)

info = train_info.tolist()
stat1, stat2, train_info, train_idx = list(zip(*train_info))
stat1 = np.array(stat1)[:, None, :] # RGB
stat2 = np.array(stat2)[:, None, :] # Gray
train_info = np.array(train_info)[:, None]
train_stat = np.concatenate([stat1, stat2], axis=1)
##
info = eval_info.tolist()
stat1, stat2, eval_info, eval_idx = list(zip(*train_info))
stat1 = np.array(stat1)[:, None, :] # RGB
stat2 = np.array(stat2)[:, None, :] # Gray
eval_info = np.array(eval_info)[:, None]
eval_stat = np.concatenate([stat1, stat2], axis=1)

##

# [dataset name][dataset inference info]
info_dict = {
    'train' : {
            'info'  : [train_stat, train_info, train_idx],
            'label' : {0: 'BN'},
            'color' : 'lightcoral'
        },        
    'infer' : {
            'info'  : [eval_stat, eval_info, eval_idx],
            # 'info'  : [data_stat, data_info, data_idx],
            # [list of expected classes] names must be 
            # aligned with indices encoded in the `info`
            'label' : {0: 'BN', 1: 'WD', 2: 'MD', 3: 'PD'},
            'color' : 'lightblue'
        },
}
# names must be aligned with indices which are recorded in the data_stat
metric_titles = ['CC', 'MI', 'SSIM']

SMALL_SIZE = 16
MEDIUM_SIZE = 40
BIGGER_SIZE = 48
####
plt.rc('axes', linewidth=2) 
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
make_boxplot(info_dict, metric_titles)
fig = plt.gcf()
fig.set_size_inches(22, 18)
fig.savefig('test.png', dpi=300)
