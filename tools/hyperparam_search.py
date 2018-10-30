import argparse
import logging

from gqcnn import GQCNNSearch
from autolab_core import YamlConfig

if __name__ == '__main__':
    # set up logger
    logging.getLogger().setLevel(logging.DEBUG) 

    # parse args
    parser = argparse.ArgumentParser(description='Hyper-parameter search for GQ-CNN.')
    parser.add_argument('dataset', type=str, default=None, help='path to dataset')
    parser.add_argument('--train_config', type=str, default='cfg/train.yaml', help='path to training config')
    parser.add_argument('--analysis_config', type=str, default='cfg/tools/analyze_gqcnn_performance.yaml')
    parser.add_argument('--split_name', type=str, default='image_wise', help='dataset split to use')
    parser.add_argument('--output_dir', type=str, default='models', help='path to store search data')
    parser.add_argument('--search_name', type=str, default=None, help='name of search')
    parser.add_argument('--cpu_cores', nargs='+', default=[], help='CPU cores to use')
    parser.add_argument('--gpu_devices', nargs='+', default=[], help='GPU devices to use')
    args = parser.parse_args()
    dataset = args.dataset
    train_config_filename = args.train_config
    analysis_config_filename = args.analysis_config
    split_name = args.split_name
    output_dir = args.output_dir
    search_name = args.search_name
    cpu_cores =[int(core) for core in args.cpu_cores]
    gpu_devices = [int(device) for device in args.gpu_devices]

    search = GQCNNSearch(YamlConfig(analysis_config_filename), [YamlConfig(train_config_filename)], [dataset], [split_name], output_dir=output_dir, search_name=search_name, cpu_cores=cpu_cores, gpu_devices=gpu_devices)
    search.search()
