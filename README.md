# [Unsupervised Tumor Characterization via Conditional Generative Adversarial Networks](https://ieeexplore.ieee.org/document/9090975)


## Package Requirement

```
pytorch 1.4.0
imgaug 0.4.0
scikit-image 0.14.2
matplotlib 3.0.2
numpy 1.15.4
opencv-python 4.1.2.30
```

## Usage Guideline

- `dataset.py` defines how the program will receive the data. Use the `ColonDataset` as template and modify the internal logic accordingly to adapt to your data.
- `config.py` contains the general running configuration (#thread, saving locations), for the network running options, please refer to `model/opt.py`
- `trainer.py` and `inferer.py` are the running scripts accordingly.
- `stats/get_patch_stat.py` contains the code for calculation of all statistics reported in the paper. 
- `plots.py` is script to plot/parse the .npy output by inferer.py to figure.

## Citation

If any part of this code is used, please give appropriate citation to our paper. <br />

BibTex entry: <br />
```
@ARTICLE{9090975,
  author={Q. D. {Vu} and K. {Kim} and J. T. {Kwak}},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Unsupervised Tumor Characterization via Conditional Generative Adversarial Networks}, 
  year={2020},
  volume={},
  number={},
  pages={1-1},}
```

## Authors

* [Quoc Dang Vu](https://github.com/vqdang)

## Acknowledgement

Thanks https://github.com/eriklindernoren/PyTorch-GAN for the collections of GAN implementations in pytorch which we are inpsired by.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details