"""
Grasp quality functions: suction quality function and parallel jaw grasping quality fuction.
Author: Jason Liu and Jeff Mahler
"""
from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import logging
import IPython
from time import time

import scipy.ndimage.filters as snf

from autolab_core import Point, PointCloud, RigidTransform
from gqcnn import Grasp2D, SuctionPoint2D, GQCNN, InputDataMode
from perception import RgbdImage, CameraIntrinsics, PointCloudImage, ColorImage, BinaryImage

# constant for display
FIGSIZE = 16

class GraspQualityFunction(object):
    """Abstract grasp quality class. """
    __metaclass__ = ABCMeta

    def __call__(self, state, actions, params=None):
        """ Evaluates grasp quality for a set of actions given a state. """
        return self.quality(state, actions, params)

    @abstractmethod
    def quality(self, state, actions, params):
        """ Evaluates grasp quality for a set of actions given a state.

        Parameters
        ----------
        state : :obj:`object`
            state of the world e.g. image
        actions : :obj:`list`
            list of actions to evaluate e.g. parallel-jaw or suction grasps
        params : :obj:`dict`
            optional parameters for the evaluation

        Returns
        -------
        :obj:`numpy.ndarray`
            vector containing the real-valued grasp quality for each candidate
        """
        pass

class SuctionQualityFunction(GraspQualityFunction):
    """Abstract wrapper class for suction quality functions (only image based metrics for now). """

    def __init__(self, config):
        """Create a suction quality function. """
        # read parameters
        self._window_size = config['window_size']
        self._sample_rate = config['sample_rate']

    def _points_in_window(self, point_cloud_image, action, segmask=None):
        """Retrieve all points on the object in a box of size self._window_size. """
        # read indices
        im_shape = point_cloud_image.shape
        i_start = int(max(action.center.y-self._window_size/2, 0))
        j_start = int(max(action.center.x-self._window_size/2, 0))
        i_end = int(min(i_start+self._window_size, im_shape[0]))
        j_end = int(min(j_start+self._window_size, im_shape[1]))
        step = int(1 / self._sample_rate)

        # read 3D points in the window
        points = point_cloud_image[i_start:i_end:step, j_start:j_end:step]
        stacked_points = points.reshape(points.shape[0]*points.shape[1], -1)

        # form the matrices for plane-fitting
        return stacked_points

    def _points_to_matrices(self, points):
        """ Convert a set of 3D points to an A and b matrix for regression. """
        A = points[:, [0, 1]]
        ones = np.ones((A.shape[0], 1))
        A = np.hstack((A, ones))
        b = points[:, 2]
        return A, b

    def _best_fit_plane(self, A, b):
        """Find a best-fit plane of points. """
        try:
            w, _, _, _ = np.linalg.lstsq(A, b) 
        except np.linalg.LinAlgError:
            logging.warning('Could not find a best-fit plane!')
            raise
        return w

    def _sum_of_squared_residuals(self, w, A, z):
        """ Returns the sum of squared residuals from the plane. """
        return np.square(np.linalg.norm(np.dot(A, w) - z))

class BestFitPlanaritySuctionQualityFunction(SuctionQualityFunction):
    """A best-fit planarity suction metric. """

    def __init__(self, config):
        """Create a best-fit planarity suction metric. """
        SuctionQualityFunction.__init__(self, config)

    def quality(self, state, actions, params=None): 
        """Given a suction point, compute a score based on a best-fit 3D plane of the neighboring points.

        Parameters
        ----------
        state : :obj:`RgbdImageState`
            An RgbdImageState instance that encapsulates rgbd_im, camera_intr, segmask, full_observed.
        action: :obj:`SuctionPoint2D`
            A suction grasp in image space that encapsulates center, approach direction, depth, camera_intr.
        params: dict
            Stores params used in computing suction quality. 

        Returns
        -------
        :obj:`numpy.ndarray`
            Array of the quality for each grasp
        """
        qualities = []

        # deproject points
        point_cloud_image = state.camera_intr.deproject_to_image(state.rgbd_im.depth)

        # compute negative SSE from the best fit plane for each grasp
        for i, action in enumerate(actions):
            if not isinstance(action, SuctionPoint2D):
                raise ValueError('This function can only be used to evaluate suction quality')
            
            points = self._points_in_window(point_cloud_image, action, segmask=state.segmask) # x,y in matrix A and z is vector z.
            A, b = self._points_to_matrices(points)
            w = self._best_fit_plane(A, b) # vector w w/ a bias term represents a best-fit plane.

            if params is not None and params['vis']['plane']:
                from visualization import Visualizer3D as vis
                mid_i = A.shape[0] / 2
                pred_z = A.dot(w)
                p0 = np.array([A[mid_i,0], A[mid_i,1], pred_z[mid_i]])
                n = np.array([w[0], w[1], -1])
                n = n / np.linalg.norm(n)
                tx = np.array([n[1], -n[0], 0])
                tx = tx / np.linalg.norm(tx)
                ty = np.cross(n, tx)
                R = np.array([tx, ty, n]).T
                T_table_world = RigidTransform(rotation=R,
                                               translation=p0,
                                               from_frame='patch',
                                               to_frame='world')
                
                vis.figure()
                vis.points(point_cloud_image.to_point_cloud(), scale=0.0025, subsample=10, random=True, color=(0,0,1))
                vis.points(PointCloud(points.T), scale=0.0025, color=(1,0,0))
                vis.table(T_table_world, dim=0.01)
                vis.show()
                
                from gqcnn import Visualizer as vis2d
                vis2d.figure()
                vis2d.imshow(state.rgbd_im.depth)
                vis2d.scatter(action.center.x, action.center.y, s=50, c='b')
                vis2d.show()

            quality = -self._sum_of_squared_residuals(w, A, b) # evaluate how well best-fit plane describles all points in window.            
            qualities.append(quality)

        return np.array(qualities)

class ApproachPlanaritySuctionQualityFunction(SuctionQualityFunction):
    """A approach planarity suction metric. """

    def __init__(self, config):
        """Create approach planarity suction metric. """
        SuctionQualityFunction.__init__(self, config)

    def _action_to_plane(self, point_cloud_image, action):
        """Convert a plane from point-normal form to general form. """
        x = int(action.center.x)
        y = int(action.center.y)
        p_0 = point_cloud_image[y, x]
        n = -action.axis
        w = np.array([-n[0]/n[2], -n[1]/n[2], np.dot(n,p_0)/n[2]])
        return w

    def quality(self, state, actions, params=None): 
        """Given a suction point, compute a score based on a best-fit 3D plane of the neighboring points.

        Parameters
        ----------
        state : :obj:`RgbdImageState`
            An RgbdImageState instance that encapsulates rgbd_im, camera_intr, segmask, full_observed.
        action: :obj:`SuctionPoint2D`
            A suction grasp in image space that encapsulates center, approach direction, depth, camera_intr.
        params: dict
            Stores params used in computing suction quality.

        Returns
        -------
        :obj:`numpy.ndarray`
            Array of the quality for each grasp
        """
        qualities = []

        # deproject points
        point_cloud_image = state.camera_intr.deproject_to_image(state.rgbd_im.depth)

        # compute negative SSE from the best fit plane for each grasp
        for i, action in enumerate(actions):
            if not isinstance(action, SuctionPoint2D):
                raise ValueError('This function can only be used to evaluate suction quality')
            
            points = self._points_in_window(point_cloud_image, action, segmask=state.segmask) # x,y in matrix A and z is vector z.
            A, b = self._points_to_matrices(points)
            w = self._action_to_plane(point_cloud_image, action) # vector w w/ a bias term represents a best-fit plane.

            if params is not None and params['vis']['plane']:
                from visualization import Visualizer3D as vis
                mid_i = A.shape[0] / 2
                pred_z = A.dot(w)
                p0 = np.array([A[mid_i,0], A[mid_i,1], pred_z[mid_i]])
                n = np.array([w[0], w[1], -1])
                n = n / np.linalg.norm(n)
                tx = np.array([n[1], -n[0], 0])
                tx = tx / np.linalg.norm(tx)
                ty = np.cross(n, tx)
                R = np.array([tx, ty, n]).T

                c = state.camera_intr.deproject_pixel(action.depth, action.center)
                d = Point(c.data - 0.01*action.axis, frame=c.frame)

                T_table_world = RigidTransform(rotation=R,
                                               translation=p0,
                                               from_frame='patch',
                                               to_frame='world')
                
                vis.figure()
                vis.points(point_cloud_image.to_point_cloud(), scale=0.0025, subsample=10, random=True, color=(0,0,1))
                vis.points(PointCloud(points.T), scale=0.0025, color=(1,0,0))
                vis.points(c, scale=0.005, color=(1,1,0))
                vis.points(d, scale=0.005, color=(1,1,0))
                vis.table(T_table_world, dim=0.01)
                vis.show()
                
                from gqcnn import Visualizer as vis2d
                vis2d.figure()
                vis2d.imshow(state.rgbd_im.depth)
                vis2d.scatter(action.center.x, action.center.y, s=50, c='b')
                vis2d.show()

            quality = -self._sum_of_squared_residuals(w, A, b) # evaluate how well best-fit plane describles all points in window.            
            qualities.append(quality)

        return np.array(qualities)

class GaussianCurvatureSuctionQualityFunction(SuctionQualityFunction):
    """A approach planarity suction metric. """

    def __init__(self, config):
        """Create approach planarity suction metric. """
        SuctionQualityFunction.__init__(self, config)

    def _points_to_matrices(self, points):
        """ Convert a set of 3D points to an A and b matrix for regression. """
        x = points[:,0]
        y = points[:,1]
        A = np.c_[x, y, x*x, x*y, y*y]
        ones = np.ones([A.shape[0], 1])
        A = np.c_[A, ones]
        b = points[:, 2]
        return A, b

    def quality(self, state, actions, params=None): 
        """Given a suction point, compute a score based on a best-fit 3D plane of the neighboring points.

        Parameters
        ----------
        state : :obj:`RgbdImageState`
            An RgbdImageState instance that encapsulates rgbd_im, camera_intr, segmask, full_observed.
        action: :obj:`SuctionPoint2D`
            A suction grasp in image space that encapsulates center, approach direction, depth, camera_intr.
        params: dict
            Stores params used in computing suction quality.

        Returns
        -------
        :obj:`numpy.ndarray`
            Array of the quality for each grasp
        """
        qualities = []

        # deproject points
        point_cloud_image = state.camera_intr.deproject_to_image(state.rgbd_im.depth)

        # compute negative SSE from the best fit plane for each grasp
        for i, action in enumerate(actions):
            if not isinstance(action, SuctionPoint2D):
                raise ValueError('This function can only be used to evaluate suction quality')
            
            points = self._points_in_window(point_cloud_image, action, segmask=state.segmask) # x,y in matrix A and z is vector z.
            A, b = self._points_to_matrices(points)
            w = self._best_fit_plane(A, b) # vector w w/ a bias term represents a best-fit plane.
            
            # compute curvature
            fx = w[0]
            fy = w[1]
            fxx = 2 * w[2]
            fxy = w[3]
            fyy = 2 * w[4]
            curvature = (fxx * fyy - fxy**2) / ((1 + fx**2 + fy**2)**2)

            # store quality
            quality = -np.abs(curvature)
            qualities.append(quality)

        return np.array(qualities)

        """
        qualities = []

        # deproject points
        point_cloud_image = state.camera_intr.deproject_to_image(state.rgbd_im.depth)
        smooth_depth_im = state.rgbd_im.depth.apply(snf.gaussian_filter,
                                                    sigma=self._depth_smooth_sigma)
        Gx, Gy = np.gradient(smooth_depth_im.data)
        Gxx, Gxy = np.gradient(Gx)
        _, Gyy = np.gradient(Gy)

        # compute Gaussian curvature for each point
        for i, action in enumerate(actions):
            if not isinstance(action, SuctionPoint2D):
                raise ValueError('This function can only be used to evaluate suction quality')
            
            # compute curvature
            v = int(action.center.x)
            u = int(action.center.y)
            gxx = Gxx[u,v]
            gyy = Gyy[u,v]
            gxy = Gxy[u,v]
            gx = Gx[u,v]
            gy = Gy[u,v]
            curvature = (gxx * gyy - gxy**2) / (1 + gx**2 + gy**2)**2

            # add qualities
            qualities.append(-curvature)

        return np.array(qualities)
        """

class GQCnnQualityFunction(GraspQualityFunction):
    def __init__(self, config):
        """Create a GQCNN suction quality function. """
        # store parameters
        self._config = config
        self._gqcnn_model_dir = config['gqcnn_model']
        self._crop_height = config['crop_height']
        self._crop_width = config['crop_width']

        # init GQ-CNN
        self._gqcnn = GQCNN.load(self._gqcnn_model_dir)

        # open tensorflow session for gqcnn
        self._gqcnn.open_session()

    def __del__(self):
        try:
            self._gqcnn.close_session()
        except:
            pass
        del self

    @property
    def gqcnn(self):
        """ Returns the GQ-CNN. """
        return self._gqcnn

    @property
    def config(self):
        """ Returns the GQCNN suction quality function parameters. """
        return self._config

    def grasps_to_tensors(self, grasps, state):
        """Converts a list of grasps to an image and pose tensor
        for fast grasp quality evaluation.

        Attributes
        ----------
        grasps : :obj:`list` of :obj:`object`
            list of image grasps to convert
        state : :obj:`RgbdImageState`
            RGB-D image to plan grasps on

        Returns
        -------
        image_arr : :obj:`numpy.ndarray`
            4D numpy tensor of image to be predicted
        pose_arr : :obj:`numpy.ndarray`
            2D numpy tensor of depth values
        """
        # parse params
        gqcnn_im_height = self.gqcnn.im_height
        gqcnn_im_width = self.gqcnn.im_width
        gqcnn_num_channels = self.gqcnn.num_channels
        gqcnn_pose_dim = self.gqcnn.pose_dim
        input_data_mode = self.gqcnn.input_data_mode
        num_grasps = len(grasps)
        depth_im = state.rgbd_im.depth

        # allocate tensors
        tensor_start = time()
        image_tensor = np.zeros([num_grasps, gqcnn_im_height, gqcnn_im_width, gqcnn_num_channels])
        pose_tensor = np.zeros([num_grasps, gqcnn_pose_dim])
        scale = float(gqcnn_im_height) / self._crop_height
        depth_im_scaled = depth_im.resize(scale)
        for i, grasp in enumerate(grasps):
            translation = scale * np.array([depth_im.center[0] - grasp.center.data[1],
                                            depth_im.center[1] - grasp.center.data[0]])
            im_tf = depth_im_scaled
            im_tf = depth_im_scaled.transform(translation, grasp.angle)
            im_tf = im_tf.crop(gqcnn_im_height, gqcnn_im_width)
            image_tensor[i,...] = im_tf.raw_data
            
            if input_data_mode == InputDataMode.TF_IMAGE:
                pose_tensor[i] = grasp.depth
            elif input_data_mode == InputDataMode.TF_IMAGE_PERSPECTIVE:
                pose_tensor[i,...] = np.array([grasp.depth, grasp.center.x, grasp.center.y])
            elif input_data_mode == InputDataMode.TF_IMAGE_SUCTION:
                pose_tensor[i,...] = np.array([grasp.depth, grasp.approach_angle])
            else:
                raise ValueError('Input data mode %s not supported' %(input_data_mode))
        logging.debug('Tensor conversion took %.3f sec' %(time()-tensor_start))
        return image_tensor, pose_tensor

    def quality(self, state, actions, params): 
        """ Evaluate the quality of a set of actions according to a GQ-CNN.

        Parameters
        ----------
        state : :obj:`RgbdImageState`
            state of the world described by an RGB-D image
        actions: :obj:`object`
            set of grasping actions to evaluate
        params: dict
            optional parameters for quality evaluation

        Returns
        -------
        :obj:`list` of float
            real-valued grasp quality predictions for each action, between 0 and 1
        """
        # form tensors
        image_tensor, pose_tensor = self.grasps_to_tensors(actions, state)
        if params is not None and params['vis']['tf_images']:
            # read vis params
            k = self.config['vis']['k']
            d = utils.sqrt_ceil(k)

            # display grasp transformed images
            vis.figure(size=(FIGSIZE,FIGSIZE))
            for i, image_tf in enumerate(image_tensor[:k,...]):
                depth = pose_tensor[i][0]
                vis.subplot(d,d,i+1)
                vis.imshow(DepthImage(image_tf))
                vis.title('Image %d: d=%.3f' %(i, depth))
            vis.show()

        # predict grasps
        predict_start = time()
        output_arr = self.gqcnn.predict(image_tensor, pose_tensor)
        q_values = output_arr[:,-1]
        logging.debug('Prediction took %.3f sec' %(time()-predict_start))
        return q_values.tolist()

class GraspQualityFunctionFactory(object):
    """Factory for grasp quality functions. """
    @staticmethod
    def quality_function(metric_type, config):
        if metric_type == 'best_fit_planarity':
            return BestFitPlanaritySuctionQualityFunction(config)
        elif metric_type == 'approach_planarity':
            return ApproachPlanaritySuctionQualityFunction(config)
        elif metric_type == 'gaussian_curvature':
            return GaussianCurvatureSuctionQualityFunction(config)
        elif metric_type == 'gqcnn':
            return GQCnnQualityFunction(config)
        else:
            raise ValueError('Grasp function type %s not supported!' %(metric_type))
