"""
Various helper functions for handling data, denoising, and computing data metrics.
Author: Vishal Satish
"""
import numpy as np
import cv2
import scipy.stats as ss
import skimage.draw as sd
import scipy.ndimage.morphology as snm
import scipy.misc as sm
import scipy.ndimage.filters as sf
import skimage.restoration as sr

from enums import InputPoseMode, InputGripperMode, DenoisingMethods

def parse_pose_data(pose_arr, input_pose_mode):
    """ Parse the given pose data according to the specified input_pose_mode

    Parameters
    ----------
    pose_arr: :obj:`ndArray`
        full pose data
    input_pose_mode: :enum:`InputPoseMode`
        enum for input pose mode, see enums.py for all
        possible input pose modes

    Returns
    -------
    :obj:`ndArray`
        parsed pose data corresponding to input pose mode
    """
    if len(pose_arr.shape) == 1:
        # wrap so that pose_arr.dim=2
        pose_arr = np.asarray([pose_arr])
    if input_pose_mode == InputPoseMode.TF_IMAGE:
        # depth
        return pose_arr[:, 2:3]
    elif input_pose_mode == InputPoseMode.TF_IMAGE_PERSPECTIVE:
        # depth, cx, cy
        return np.c_[pose_arr[:, 2:3], pose_arr[:, 4:6]]
    elif input_pose_mode == InputPoseMode.RAW_IMAGE:
        # u, v, depth, theta
        return pose_arr[:, :4]
    elif input_pose_mode == InputPoseMode.RAW_IMAGE_PERSPECTIVE:
        # u, v, depth, theta, cx, cy
        return pose_arr[:, :6]
    elif input_pose_mode == InputPoseMode.TF_IMAGE_SUCTION:
        # depth, theta
        return pose_arr[:, 2:4]
    else:
        raise ValueError('Input pose mode {} not supported'.format(input_pose_mode))

def parse_gripper_data(gripper_arr, input_gripper_mode):
    if len(gripper_arr.shape) == 1:
        # wrap so that gripper_arr.dim=2
        gripper_arr = np.asarray([gripper_arr])
    if input_gripper_mode == InputGripperMode.WIDTH:
        # width
        return gripper_arr[:, 0:1]
    elif input_gripper_mode == InputGripperMode.ALL:
        # width, palm_depth, fingertip_x, fingertip_y
        return gripper_arr
    else:
        raise ValueError('Input gripper mode {} not supportd'.format(input_gripper_mode))

def compute_data_metrics(experiment_dir, data_dir, im_height, im_width, total_pose_elems, input_pose_mode, 
    train_index_map, im_filenames, pose_filenames, gripper_param_filenames=None, gripper_depth_mask_filenames=None, 
    total_gripper_param_elems=None, num_random_files=100):
    """ Compute mean & std for images, poses, and possible gripper_depth_masks, gripper_params"""
    num_files = len(im_filenames)

    # compute image mean
    logging.info('Computing image metrics')
    im_mean = 0
    im_var = 0 
    random_file_indices = np.random.choice(num_files, size=num_random_files, replace=False)
    num_summed = 0
    for k in random_file_indices.tolist():
        im_filename = im_filenames[k]
        im_data = np.load(os.path.join(data_dir, im_filename))['arr_0']
        im_mean += np.sum(im_data[train_index_map[im_filename], ...])
        num_summed += im_data[train_index_map[im_filename], ...].shape[0]
    im_mean = im_mean / (num_summed * im_height * im_width)

    for k in random_file_indices.tolist():
        im_filename = im_filenames[k]
        im_data = np.load(os.path.join(data_dir, im_filename))['arr_0']
        im_var += np.sum((im_data[train_index_map[im_filename], ...] - im_mean)**2)
    im_std = np.sqrt(im_var / (num_summed * im_height * im_width))

    # compute pose mean
    logging.info('Computing pose metrics')
    pose_mean = np.zeros(total_pose_elems)
    pose_var = np.zeros(total_pose_elems)
    num_summed = 0
    random_file_indices = np.random.choice(num_files, size=num_random_files, replace=False)
    for k in random_file_indices.tolist():
        im_filename = im_filenames[k]
        pose_filename = pose_filenames[k]
        pose_data = np.load(os.path.join(data_dir, pose_filename))['arr_0']
        if input_pose_mode == InputPoseMode.TF_IMAGE_SUCTION:
            rand_indices = np.random.choice(pose_data.shape[0], size=pose_data.shape[0] / 2, replace=False)
            pose_data[rand_indices, 3] = -pose_data[rand_indices, 3]
        pose_data = pose_data[train_index_map[im_filename], :]
        pose_mean += np.sum(pose_data, axis=0)
        num_summed += pose_data.shape[0]
    pose_mean = pose_mean / num_summed

    for k in random_file_indices.tolist():
        im_filename = im_filenames[k]
        pose_filename = pose_filenames[k]
        pose_data = np.load(os.path.join(data_dir, pose_filename))['arr_0']
        if input_pose_mode == InputPoseMode.TF_IMAGE_SUCTION:
            rand_indices = np.random.choice(pose_data.shape[0], size=pose_data.shape[0] / 2, replace=False)
            pose_data[rand_indices, 3] = -pose_data[rand_indices, 3]
        pose_data = pose_data[train_index_map[im_filename], :]
        pose_var += np.sum((pose_data - pose_mean)**2, axis=0)
    pose_std = np.sqrt(pose_var / num_summed)

    # std cannot be 0
    pose_std[np.where(pose_std==0)] = 1.0

    # compute gripper param mean
    gripper_mean = None
    gripper_std = None
    if gripper_param_filenames:
        logging.info('Computing gripper parameter metrics')
        gripper_mean = np.zeros(total_gripper_param_elems)
        gripper_var = np.zeros(total_gripper_param_elems)
        num_summed = 0
        random_file_indices = np.random.choice(num_files, size=num_random_files, replace=False)
        for k in random_file_indices.tolist():
            im_filename = im_filenames[k]
            gripper_filename = gripper_param_filenames[k]
            gripper_data = np.load(os.path.join(data_dir, gripper_filename))['arr_0']
            gripper_data = gripper_data[train_index_map[im_filename], :]
            gripper_mean += np.sum(gripper_data, axis=0)
            num_summed += gripper_data.shape[0]
        gripper_mean = gripper_mean / num_summed

        for k in random_file_indices.tolist():
            im_filename = im_filenames[k]
            gripper_filename = gripper_param_filenames[k]
            gripper_data = np.load(os.path.join(data_dir, gripper_filename))['arr_0']
            gripper_data = gripper_data[train_index_map[im_filename], :]
            gripper_var += np.sum((gripper_data - gripper_mean)**2, axis=0)
        gripper_std = np.sqrt(gripper_var / num_summed)
        
        # std cannot be 0
        gripper_std[np.where(gripper_std==0)] = 1.0

    # compute gripper depth mask mean
    gripper_depth_mask_mean = None
    gripper_depth_mask_std = None
    if gripper_depth_mask_filenames:
        logging.info('Computing gripper depth mask metrics')
        gripper_depth_mask_mean_filename = os.path.join(experiment_dir, 'gripper_depth_mask_mean.npy')
        gripper_depth_mask_std_filename = os.path.join(experiment_dir, 'gripper_depth_mask_std.npy')
        gripper_depth_mask_mean = np.zeros((2)) # one for each channel of mask(fingertip/palm)
        gripper_depth_mask_var = np.zeros((2))
        random_file_indices = np.random.choice(num_files, size=num_random_files, replace=False)
        num_summed = 0
        for k in random_file_indices.tolist():
            im_filename = im_filenames[k]
            mask_filename = gripper_depth_mask_filenames[k]
            mask_data = np.load(os.path.join(data_dir, mask_filename))['arr_0']
            gripper_depth_mask_mean[0] += np.sum(mask_data[train_index_map[im_filename], ..., 0])
            gripper_depth_mask_mean[1] += np.sum(mask_data[train_index_map[im_filename], ..., 1])
            num_summed += mask_data[train_index_map[im_filename], ..., 1].shape[0]
        gripper_depth_mask_mean = gripper_depth_mask_mean / (num_summed * im_height * im_width)

        for k in random_file_indices.tolist():
            im_filename = im_filenames[k]
            mask_filename = gripper_depth_mask_filenames[k]
            mask_data = np.load(os.path.join(data_dir, mask_filename))['arr_0']
            gripper_depth_mask_var[0] += np.sum((mask_data[train_index_map[im_filename], ..., 0] - gripper_depth_mask_mean[0])**2)
            gripper_depth_mask_var[1] += np.sum((mask_data[train_index_map[im_filename], ..., 1] - gripper_depth_mask_mean[1])**2)
        gripper_depth_mask_std = np.sqrt(gripper_depth_mask_var / (num_summed * im_height * im_width))

    return image_mean, image_std, pose_mean, pose_std, gripper_mean, gripper_std, gripper_depth_mask_mean, gripper_depth_mask_std

def compute_grasp_metric_stats(data_dir, im_filenames, label_filenames, val_index_map, metric_thresh):
    """ Computes min, max, mean, median statistics for grasp robustness metric. Also computes percentage of
    positive labels. """

    logging.info('Computing grasp metric stats')
    all_metrics = None
    all_val_metrics = None
    for im_filename, metric_filename in zip(im_filenames, label_filenames):
        metric_data = np.load(os.path.join(data_dir, metric_filename))['arr_0']
        indices = val_index_map[im_filename]
        val_metric_data = metric_data[indices]
        if all_metrics is None:
            all_metrics = metric_data
        else:
            all_metrics = np.r_[all_metrics, metric_data]
        if all_val_metrics is None:
            all_val_metrics = val_metric_data
        else:
            all_val_metrics = np.r_[all_val_metrics, val_metric_data]
    min_metric = np.min(all_metrics)
    max_metric = np.max(all_metrics)
    mean_metric = np.mean(all_metrics)
    median_metric = np.median(all_metrics)

    pct_pos_val = float(np.sum(all_val_metrics > metric_thresh)) / all_val_metrics.shape[0]
    
    return min_metric, max_metric, mean_metric, median_metric, pct_pos_val

def denoise(self, im_arr, im_height, im_width, im_channels, denoising_params, pose_arr=None, pose_dim=1, only_dropout=False, mask_and_inpaint=False):
    """ Adds noise to a batch of images and possibly poses """
    
    # make deepcopy of input arrays
    im_arr = copy.deepcopy(im_arr)
    if pose_arr:
        pose_arr = copy.deepcopy(pose_arr)

    # apply denoising
    for method, params in denoising_params.iteritems():
        # multiplicative denoising
        if method == DenoisingMethods.MULTIPLICATIVE_DENOISING and not only_dropout:
            mult_samples = ss.gamma.rvs(params['gamma_shape'], scale=params['gamma_scale'], size=len(im_arr))
            mult_samples = mult_samples[:,np.newaxis,np.newaxis,np.newaxis]
            im_arr = im_arr * np.tile(mult_samples, [1, im_height, im_width, im_channels])

        # randomly dropout regions of the image for robustness
        elif method == DenoisingMethods.IMAGE_DROPOUT and not only_dropout:
            for i in range(len(im_arr)):
                if np.random.rand() < params['image_dropout_rate']:
                    image = im_arr[i,:,:,0]
                    nonzero_px = np.where(image > 0)
                    nonzero_px = np.c_[nonzero_px[0], nonzero_px[1]]
                    num_nonzero = nonzero_px.shape[0]
                    num_dropout_regions = ss.poisson.rvs(params['dropout_poisson_mean']) 
                    
                    # sample ellipses
                    dropout_centers = np.random.choice(num_nonzero, size=num_dropout_regions)
                    x_radii = ss.gamma.rvs(params['dropout_radius_shape'], scale=params['dropout_radius_scale'], size=num_dropout_regions)
                    y_radii = ss.gamma.rvs(params['dropout_radius_shape'], scale=params['dropout_radius_scale'], size=num_dropout_regions)

                    # set interior pixels to zero
                    for j in range(num_dropout_regions):
                        ind = dropout_centers[j]
                        dropout_center = nonzero_px[ind, :]
                        x_radius = x_radii[j]
                        y_radius = y_radii[j]
                        dropout_px_y, dropout_px_x = sd.ellipse(dropout_center[0], dropout_center[1], y_radius, x_radius, shape=image.shape)
                        image[dropout_px_y, dropout_px_x] = 0.0
                    im_arr[i,:,:,0] = image

        # dropout a region around the areas of the image with high gradient
        elif method == DenoisingMethods.GRADIENT_DROPOUT and not only_dropout:
            for i in range(len(im_arr)):
                if np.random.rand() < params['gradient_dropout_rate']:
                    image = im_arr[i,:,:,0]
                    grad_mag = sf.gaussian_gradient_magnitude(image, sigma=params['gradient_dropout_sigma'])
                    thresh = ss.gamma.rvs(params['gradient_dropout_shape'], params['gradient_dropout_scale'], size=1)
                    high_gradient_px = np.where(grad_mag > thresh)
                    image[high_gradient_px[0], high_gradient_px[1]] = 0.0
                im_arr[i,:,:,0] = image

        # add correlated Gaussian noise
        elif method == DenoisingMethods.GAUSSIAN_PROCESS_DENOISING and not only_dropout:
            for i in range(len(im_arr)):
                if np.random.rand() < params['gaussian_process_rate']:
                    image = im_arr[i,:,:,0]
                    gp_noise = ss.norm.rvs(scale=params['gp_sigma'], size=params['gp_num_pix']).reshape(params['gp_sample_height'], params['gp_sample_width'])
                    gp_noise = sm.imresize(gp_noise, params['gp_rescale_factor'], interp='bicubic', mode='F')
                    image[image > 0] += gp_noise[image > 0]
                    im_arr[i,:,:,0] = image

        # run open and close filters
        elif method == DenoisingMethods.MORPHOLOGICAL and not only_dropout:
            for i in range(len(im_arr)):
                image = im_arr[i,:,:,0]
                sample = np.random.rand()
                morph_filter_dim = ss.poisson.rvs(params['morph_poisson_mean'])                         
                if sample < params['morph_open_rate']:
                    image = snm.grey_opening(image, size=morph_filter_dim)
                else:
                    closed_train_image = snm.grey_closing(image, size=morph_filter_dim)
                    
                    # set new closed pixels to the minimum depth, mimicing the table
                    new_nonzero_px = np.where((image == 0) & (closed_train_image > 0))
                    closed_train_image[new_nonzero_px[0], new_nonzero_px[1]] = np.min(image[image>0])
                    image = closed_train_image.copy()
                im_arr[i,:,:,0] = image                        

        # randomly dropout borders of the image for robustness
        elif method == DenoisingMethods.BORDER_DISTORTION:
            for i in range(len(im_arr)):
                if np.random.rand() < params['border_distortion_rate']:
                    image = im_arr[i,:,:,0]
                    original = image.copy()
                    mask = np.zeros(image.shape)
                    grad_mag = sf.gaussian_gradient_magnitude(image, sigma=params['border_grad_sigma'])
                    high_gradient_px = np.where(grad_mag > params['border_grad_thresh'])
                    high_gradient_px = np.c_[high_gradient_px[0], high_gradient_px[1]]
                    num_nonzero = high_gradient_px.shape[0]
                    if num_nonzero == 0:
                        continue
                    num_dropout_regions = ss.poisson.rvs(params['border_poisson_mean']) 

                    # sample ellipses
                    dropout_centers = np.random.choice(num_nonzero, size=num_dropout_regions)
                    x_radii = ss.gamma.rvs(params['border_radius_shape'], scale=params['border_radius_scale'], size=num_dropout_regions)
                    y_radii = ss.gamma.rvs(params['border_radius_shape'], scale=params['border_radius_scale'], size=num_dropout_regions)

                    # set interior pixels to zero or one
                    for j in range(num_dropout_regions):
                        ind = dropout_centers[j]
                        dropout_center = high_gradient_px[ind, :]
                        x_radius = x_radii[j]
                        y_radius = y_radii[j]
                        dropout_px_y, dropout_px_x = sd.ellipse(dropout_center[0], dropout_center[1], y_radius, x_radius, shape=image.shape)
                    
                        if params['border_fill_type'] == 'zero':
                            image[dropout_px_y, dropout_px_x] = 0.0
                            mask[dropout_px_y, dropout_px_x] = 1
                        elif params['border_fill_type'] == 'inf':
                            image[dropout_px_y, dropout_px_x] = np.inf
                            mask[dropout_px_y, dropout_px_x] = np.inf
                        elif params['border_fill_type'] == 'machine_max':
                            image[dropout_px_y, dropout_px_x] = np.finfo(np.float64).max
                            mask[dropout_px_y, dropout_px_x] = np.finfo(np.float64).max
                
                    if mask_and_inpaint:
                        image = sr.inpaint.inpaint_biharmonic(image, mask)
                        image = image.reshape((32, 32, 1))
                        mask = mask.reshape((32, 32, 1))
                        image = np.c_[image, mask]
                        inpainted_image = image[:, :, 0]
                        mask = image[:, :, 1]
                        im_arr[i] = image
                    else: 
                        im_arr[i,:,:,0] = image

        # randomly replace background pixels with constant depth
        elif method == DenoisingMethods.BACKROUND_DENOISING and not only_dropout:
            for i in range(len(im_arr)):
                image = im_arr[i,:,:,0]                
                if np.random.rand() < params['background_rate']:
                    image[image>0] = params['background_min_depth'] + (params['background_max_depth'] - params['background_min_depth']) * np.random.rand()
                im_arr[i,:,:,0] = image

        # symmetrize images and poses
        elif method == DenoisingMethods.SYMMETRIZE and not only_dropout:
            im_center = np.asarray([float(im_width - 1) / 2, float(im_height - 1) / 2])
            for i in range(len(im_arr)):
                image = im_arr[i,:,:,0]
                # rotate with 50% probability
                if np.random.rand() < 0.5:
                    theta = 180.0
                    rot_map = cv2.getRotationMatrix2D(tuple(im_center), theta, 1)
                    image = cv2.warpAffine(image, rot_map, (im_height, im_width), flags=cv2.INTER_NEAREST)
                    if pose_dim > 4:
                        pose_arr[i,4] = -pose_arr[i,4]
                        pose_arr[i,5] = -pose_arr[i,5]
                # reflect left right with 50% probability
                if np.random.rand() < 0.5:
                    image = np.fliplr(image)
                    if pose_dim > 4:
                        pose_arr[i,5] = -pose_arr[i,5]
                # reflect up down with 50% probability
                if np.random.rand() < 0.5:
                    image = np.flipud(image)
                    if pose_dim > 4:
                        pose_arr[i,4] = -pose_arr[i,4]
                im_arr[i,:,:,0] = image

    return im_arr, pose_arr
