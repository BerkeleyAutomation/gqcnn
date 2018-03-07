import collections
import os
import json

from autolab_core import YamlConfig

PATH_TO_YAML = '/home/vsatish/Workspace/dev/gqcnn/cfg/tools/train_dex-net_2.0.yaml'
JSON_OUTPUT_DIR = '/home/vsatish/Data/dexnet/data/models/grasp_quality/dex-net_2.0_image_wise'
JSON_FNAME = 'config.json'

if __name__ == '__main__':
    yaml = YamlConfig(PATH_TO_YAML)
    orderedDict = collections.OrderedDict()
    for key in yaml.keys():
        orderedDict[key] = yaml[key]
    with open(os.path.join(JSON_OUTPUT_DIR, JSON_FNAME), 'w') as outfile:
        json.dump(orderedDict, outfile)
