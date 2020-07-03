# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents).
All Rights Reserved. Permission to use, copy, modify, and distribute this
software and its documentation for educational, research, and not-for-profit
purposes, without fee and without a signed licensing agreement, is hereby
granted, provided that the above copyright notice, this paragraph and the
following two paragraphs appear in all copies, modifications, and
distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150
Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
otl@berkeley.edu,
http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Utility functions for hyper-parameter search.

Author
------
Vishal Satish
"""
from collections import OrderedDict, defaultdict
import copy
from datetime import datetime
import itertools


def get_fields_to_search_over(train_config, prev_keys=[]):
    fields = []
    anchored_fields = defaultdict(list)
    for key in train_config:
        if isinstance(train_config[key], list):
            prev_keys_copy = copy.deepcopy(prev_keys)
            prev_keys_copy.append(key)
            if isinstance(train_config[key][0],
                          str) and train_config[key][0].startswith("anchor_"):
                anchored_fields[train_config[key][0]].append(prev_keys_copy)
                train_config[key] = train_config[key][1:]
            else:
                fields.append(prev_keys_copy)
        elif isinstance(train_config[key], OrderedDict):
            prev_keys_copy = copy.deepcopy(prev_keys)
            prev_keys_copy.append(key)
            sub_fields, sub_anchored_fields = get_fields_to_search_over(
                train_config[key], prev_keys=prev_keys_copy)
            fields.extend(sub_fields)
            update_dict(anchored_fields, sub_anchored_fields)
    return fields, anchored_fields


def update_dict(dict1, dict2):
    for key, val in dict2.items():
        if key in dict1:
            dict1[key].extend(val)
        else:
            dict1[key] = val


def get_nested_key(cfg, key):
    val = cfg
    for k in key:
        val = val[k]
    return val


def set_nested_key(cfg, key, val):
    root_field = cfg
    for k in key[:-1]:
        root_field = root_field[k]
    root_field[key[-1]] = val


def gen_config_summary_dict(hyperparam_combination):
    summary_dict = {}
    for key, val in hyperparam_combination:
        summary_dict["/".join(key)] = val
    return summary_dict


def parse_master_train_config(train_config):
    configs = []
    hyperparam_search_fields, hyperparam_anchored_search_fields = \
        get_fields_to_search_over(train_config)

    # Ensure a one-to-one mapping between hyperparameters of fields with
    # matching anchor tags.
    for anchor_tag, fields in hyperparam_anchored_search_fields.items():
        num_params = []
        for field in fields:
            num_params.append(len(get_nested_key(train_config, field)))
        invalid_anchor_tag_msg = ("All fields in anchor tag '{}' do not have"
                                  " the same # of parameters to search over!")
        assert max(num_params) == min(
            num_params), invalid_anchor_tag_msg.format(anchor_tag)

    # If there is nothing to search over just return the given config.
    if len(hyperparam_search_fields) == 0 and len(
            hyperparam_anchored_search_fields) == 0:
        return [("", train_config)]

    # Generate a list of all the possible hyper-parameters to search over.
    # Normal fields.
    hyperparam_search_params = []
    for search_field in hyperparam_search_fields:
        search_field_params = []
        for val in get_nested_key(train_config, search_field):
            search_field_params.append((search_field, val))
        hyperparam_search_params.append(search_field_params)
    # Anchored fields.
    for anchored_fields in hyperparam_anchored_search_fields.values():
        combinations = [[] for _ in range(
            len(get_nested_key(train_config, anchored_fields[0])))]
        for field in anchored_fields:
            for idx, val in enumerate(get_nested_key(train_config, field)):
                combinations[idx].append((field, val))
        hyperparam_search_params.append(combinations)

    # Get all permutations of the possible hyper-parameters.
    hyperparam_combinations = list(
        itertools.product(*hyperparam_search_params))

    def flatten_combo(combo):
        flattened = []
        for item in combo:
            if isinstance(item, list):
                for sub_item in item:
                    flattened.append(sub_item)
            else:
                flattened.append(item)
        return flattened

    hyperparam_combinations = [
        flatten_combo(combo) for combo in hyperparam_combinations
    ]

    # Generate possible configs to search over.
    for combo in hyperparam_combinations:
        config = copy.deepcopy(train_config)
        for field, val in combo:
            set_nested_key(config, field, val)
        configs.append((gen_config_summary_dict(combo), config))
    return configs


def gen_timestamp():
    return str(datetime.now()).split(".")[0].replace(" ", "_")


def gen_trial_params_train(master_train_configs, datasets, split_names):
    trial_params = []
    for master_train_config, dataset, split_name in zip(
            master_train_configs, datasets, split_names):
        train_configs = parse_master_train_config(master_train_config)
        for i, (hyperparam_summary_dict,
                train_config) in enumerate(train_configs):
            trial_name = "{}_{}_trial_{}_{}".format(
                dataset.split("/")[-3], split_name, i, gen_timestamp())
            trial_params.append((trial_name, hyperparam_summary_dict,
                                 train_config, dataset, split_name))
    return trial_params


def gen_trial_params_finetune(master_train_configs, datasets, base_models,
                              split_names):
    trial_params = []
    for master_train_config, dataset, base_model, split_name in zip(
            master_train_configs, datasets, base_models, split_names):
        train_configs = parse_master_train_config(master_train_config)
        for i, (hyperparam_summary_dict,
                train_config) in enumerate(train_configs):
            trial_name = "{}_{}_trial_{}_{}".format(
                dataset.split("/")[-3], split_name, i, gen_timestamp())
            trial_params.append(
                (trial_name, hyperparam_summary_dict, train_config, dataset,
                 base_model, split_name))
    return trial_params


def gen_trial_params(master_train_configs,
                     datasets,
                     split_names,
                     base_models=[]):
    if len(base_models) > 0:
        return gen_trial_params_finetune(master_train_configs, datasets,
                                         base_models, split_names)
    else:
        return gen_trial_params_train(master_train_configs, datasets,
                                      split_names)


def log_trial_status(trials):
    status_str = "--------------------TRIAL STATUS--------------------"
    for trial in trials:
        status_str += "\n"
        status_str += "[{}]".format(str(trial))
    return status_str
