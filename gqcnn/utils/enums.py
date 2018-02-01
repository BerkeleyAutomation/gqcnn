"""
Class for storing constants/templates/enums 
Author: Vishal Satish
"""
import tensorflow as tf

# general constants
class GeneralConstants:
    SEED = 95417239
    timeout_option = tf.RunOptions(timeout_in_ms=1000000)

# enum for templates for setting up output directories
class OutputDirTemplates:
    MODEL_DIR = 'model'
    SUMMARY_DIR = 'tensorboard_summaries'
    FILTER_DIR = 'filters'

# enum for data split modes
class DataSplitMode:
    IMAGE_WISE = 'image_wise'
    OBJECT_WISE = 'object_wise'
    STABLE_POSE_WISE = 'stable_pose_wise'

# enum for templates for file reading
class DataFileTemplates:
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
    gripper_params_template = 'gripper_params'
    gripper_segmask_template = 'gripper_segmask'
    gripper_depth_template = 'gripper_depth'
    filename_placeholder = '-11111111111111'

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

# enum for training label preproc modes
class PreprocMode:
    NORMALIZATION = 'normalized'
    NONE = 'none'

# enum for input pose data formats
class InputPoseMode:
    TF_IMAGE = 'tf_image'
    TF_IMAGE_PERSPECTIVE = 'tf_image_with_perspective'
    RAW_IMAGE = 'raw_image'
    RAW_IMAGE_PERSPECTIVE = 'raw_image_with_perspective'
    TF_IMAGE_SUCTION = 'tf_image_suction'

# enum for input gripper data formats
class InputGripperMode:
    WIDTH = 'width'
    ALL = 'all'
    DEPTH_MASK = 'depth_mask'
    NONE = 'none'
