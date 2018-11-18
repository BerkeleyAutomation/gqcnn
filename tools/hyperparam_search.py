"""
Script for searching over Grasp Quality Convolutional Neural Network (GQ-CNN) hyper-parameters.

Author
------
Vishal Satish
"""
import argparse
import sys

from gqcnn import GQCNNSearch
from autolab_core import YamlConfig, Logger

# set up logger
logger = Logger.get_logger('tools/hyperparam_search.py') 

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Hyper-parameter search for GQ-CNN.')
    parser.add_argument('datasets', nargs='+', default=None, help='path to datasets')
    parser.add_argument('--base_model_dirs', nargs='+', default=[], help='path to pre-trained base models for fine-tuning')
    parser.add_argument('--train_configs', nargs='+', default=['cfg/train.yaml'], help='path to training configs')
    parser.add_argument('--analysis_config', type=str, default='cfg/tools/analyze_gqcnn_performance.yaml')
    parser.add_argument('--split_names', nargs='+', default=['image_wise'], help='dataset splits to use')
    parser.add_argument('--output_dir', type=str, default='models', help='path to store search data')
    parser.add_argument('--search_name', type=str, default=None, help='name of search')
    parser.add_argument('--cpu_cores', nargs='+', default=[], help='CPU cores to use')
    parser.add_argument('--gpu_devices', nargs='+', default=[], help='GPU devices to use')
    args = parser.parse_args()
    datasets = args.datasets
    base_model_dirs = args.base_model_dirs
    train_configs = args.train_configs
    analysis_config = args.analysis_config
    split_names = args.split_names
    output_dir = args.output_dir
    search_name = args.search_name
    cpu_cores =[int(core) for core in args.cpu_cores]
    gpu_devices = [int(device) for device in args.gpu_devices]

    assert len(datasets) == len(train_configs), 'Must have same number of datasets as training configs!'
    if len(base_model_dirs) > 0:
        assert len(base_model_dirs) == len(datasets), 'Must have same number of base models for fine-tuning as datasets and training configs!'
    if len(split_names) < len(datasets):
        if len(split_names) == 1:
            logger.warning('Using split "{}" for all datasets/configs...'.format(split_names[0]))
            split_names *= len(datasets)
        else:
            raise ValueError("Can't have fewer splits that datasets/configs provided unless there is only one.")

    # parse configs
    analysis_config = YamlConfig(analysis_config)
    train_configs = [YamlConfig(cfg) for cfg in train_configs]

    # search
    search = GQCNNSearch(analysis_config, train_configs, datasets, split_names, output_dir=output_dir, search_name=search_name, cpu_cores=cpu_cores, gpu_devices=gpu_devices, base_models=base_model_dirs)
    search.search()
