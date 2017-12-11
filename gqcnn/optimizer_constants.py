"""
Class for storing constants/enums for the DeepOptimizer
Author: Vishal Satish
"""

# other constants
class GeneralConstants:
    SEED = 95417238

# enum for templates for file reading
class ImageFileTemplates:
    binary_im_tensor_template = 'binary_ims_raw'
    depth_im_tensor_template = 'depth_ims_raw'
    binary_im_tf_tensor_template = 'binary_ims_tf'
    color_im_tf_tensor_template = 'color_ims_tf'
    gray_im_tf_tensor_template = 'gray_ims_tf'
    depth_im_tf_tensor_template = 'depth_ims_tf'
    depth_im_tf_table_tensor_template = 'depth_ims_tf_table'
    table_mask_template = 'table_mask'
    hand_poses_template = 'hand_poses'
    object_labels_template = 'object_labels'
    pose_labels_template = 'pose_labels'

# enum for image modalities
class ImageMode:
    BINARY = 'binary'
    DEPTH = 'depth'
    BINARY_TF = 'binary_tf'
    COLOR_TF = 'color_tf'
    GRAY_TF = 'gray_tf'
    DEPTH_TF = 'depth_tf'
    DEPTH_TF_TABLE = 'depth_tf_table'

# enum for training modes
class TrainingMode:
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'

# enum for preproc
class PreprocMode:
    NORMALIZATION = 'normalized'
    NONE = 'none'

# enum for input data formats
class InputDataMode:
    TF_IMAGE = 'tf_image'
    TF_IMAGE_PERSPECTIVE = 'tf_image_with_perspective'
    RAW_IMAGE = 'raw_image'
    RAW_IMAGE_PERSPECTIVE = 'raw_image_with_perspective'
