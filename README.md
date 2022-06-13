# What Matters For Meta-Learning Vision Regression Tasks?
This is the official code for the paper "What Matters For Meta-Learning Vision Regression Tasks?" by Ning Gao et al. accepted to CVPR 2022. The code allows the users to reproduce and extend the results reported in the study. Please cite the above paper when reporting, reproducing or extending the results.

[[Arxiv](https://arxiv.org/abs/2203.04905)] [[CVPR22'](https://openaccess.thecvf.com/content/CVPR2022/html/Gao_What_Matters_for_Meta-Learning_Vision_Regression_Tasks_CVPR_2022_paper.html)]

## Purpose of the project

This software is a research prototype, solely developed for and published as part of the publication "What Matters For Meta-Learning Vision Regression Tasks?". It will neither be maintained nor monitored in any way.

## Requirements, test, install, use, etc.

### Installation
- install CUDA10.1
- clone the repo using `git lfs`
```shell
  git lfs clone https://github.com/boschresearch/what-matters-for-meta-learning.git
  cd what-matters-for-meta-learning
```
- setup environment
```shell
  conda create -n mlvr python=3.7.9 pip
  conda activate mlvr
  pip install -r requirements.txt
```
### Datasets
Extract Distractor, ShapeNet1D, ShapeNet3D datasets and background images used for ShapeNet3D from `data/`, Pascal1D dataset can be generated following prior work [Meta-Learning without Memorization](https://github.com/google-research/google-research/tree/ccc94ce348360ddcd41c749d4088d468ccfd1eaf/meta_learning_without_memorization). Extract all datasets under `data/`, datafolder should be structured as:
```shell
data/
├── distractor/
├── ShapeNet1D/
├── ShapeNet3D_azi180ele30/
├── Pascal1D/
├── bg_images.npy
dataset/
networks/
...
```

### Models
We provide three pretrained models for: distractor task with CNP + max aggregation, ANP for Shapenet1D task and Shapenet3D. Download the models from [here](https://drive.google.com/file/d/17sHI5TfMWlMyB8RPfz9OJRnQ_f2yShB5/view?usp=sharing) and extract under `results/train/`.

### Evaluation
evaluate and visualize predictions on distractor task:
```shell
python evaluate_and_plot_distractor.py --config cfg/evaluation/eval_and_plot/${config file}
```
evaluate and visualize on ShapeNet1D:
```shell
python evaluate_and_plot_shapenet1d.py --config cfg/evaluation/eval_and_plot/${config file}
```
evaluate and visualize predictions on ShapeNet3D:
```shell
python evaluate_and_plot_shapenet3d.py --config cfg/evaluation/eval_and_plot/${config file}
```
statistical evaluation on novel tasks:
```shell
python evaluation.py --config cfg/evaluation/${config file}
```
### Training
Some training config examples are provided in `cfg/train/`, you can also play around and change parameters with different combinations.

Training distractor task requires 32GB memory to load all category datasets:

`python train.py --config cfg/train/${Distractor_script}`

for example, if you want to traing ANP with DA and TA for distractor task:

`python train.py --config cfg/train/ANP_DA+TA_Distractor.yaml`

### Refine on Single Task:
Adapt the load model path in config file and run: 

`python refinement.py --config cfg/refinement/${single_task_config}`


## Citation
If you use this work please cite
```
@InProceedings{Gao_2022_CVPR,
    author    = {Gao, Ning and Ziesche, Hanna and Vien, Ngo Anh and Volpp, Michael and Neumann, Gerhard},
    title     = {What Matters for Meta-Learning Vision Regression Tasks?},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {14776-14786}
}
```

## License

what-matters-for-meta-learning is open-sourced under the AGPL-3.0 license. See the [LICENSE](LICENSE) file for details.
For a list of other open source components included in what-matters-for-meta-learning see the file [3rd-party-licenses.txt](3rd-party-licenses.txt).


