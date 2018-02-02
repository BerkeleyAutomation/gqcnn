# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Common 2D visualizations using pyplot
Author: Jeff Mahler
"""
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from autolab_core import Box, Contour
from perception import BinaryImage, ColorImage, DepthImage, GrayscaleImage, RgbdImage, GdImage, SegmentationImage

class Visualizer:
    @staticmethod
    def figure(size=(8,8), *args, **kwargs):
        """ Creates a figure.

        Parameters
        ----------
        size : 2-tuple
           size of the view window in inches
        args : list
           args of mayavi figure
        kwargs : list
           keyword args of mayavi figure

        Returns
        -------
        pyplot figure
            the current figure
        """
        return plt.figure(figsize=size, *args, **kwargs)
    
    @staticmethod
    def show(*args, **kwargs):
        """ Show the current figure """
        plt.show(*args, **kwargs)

    @staticmethod
    def clf(*args, **kwargs):
        """ Clear the current figure """
        plt.clf(*args, **kwargs)

    @staticmethod
    def savefig(*args, **kwargs):
        """ Save the current figure """
        plt.savefig(*args, **kwargs)

    @staticmethod
    def colorbar(*args, **kwargs):
        """ Adds a colorbar to the current figure """
        plt.colorbar(*args, **kwargs)

    @staticmethod
    def subplot(*args, **kwargs):
        """ Creates a subplot in the current figure """
        plt.subplot(*args, **kwargs)

    @staticmethod
    def title(*args, **kwargs):
        """ Creates a title in the current figure """
        plt.title(*args, **kwargs)

    @staticmethod
    def xlabel(*args, **kwargs):
        """ Creates an x axis label in the current figure """
        plt.xlabel(*args, **kwargs)

    @staticmethod
    def ylabel(*args, **kwargs):
        """ Creates an y axis label in the current figure """
        plt.ylabel(*args, **kwargs)

    @staticmethod
    def scatter(*args, **kwargs):
        """ Scatters points """
        plt.scatter(*args, **kwargs)

    @staticmethod
    def plot(*args, **kwargs):
        """ Plots lines """
        plt.plot(*args, **kwargs)
    
    @staticmethod
    def imshow(image, **kwargs):
        """ Displays an image.

        Parameters
        ----------
        image : :obj:`perception.Image`
            image to display
        """
        if isinstance(image, BinaryImage) or isinstance(image, GrayscaleImage):
            plt.imshow(image.data, cmap=plt.cm.gray, **kwargs)
        elif isinstance(image, ColorImage) or isinstance(image, SegmentationImage):
            plt.imshow(image.data, **kwargs)
        elif isinstance(image, DepthImage):
            plt.imshow(image.data, cmap=plt.cm.gray_r, **kwargs)
        elif isinstance(image, RgbdImage):
            # default to showing color only, for now...
            plt.imshow(image.color.data, **kwargs)
        elif isinstance(image, GdImage):
            # default to showing gray only, for now...
            plt.imshow(image.gray.data, cmap=plt.cm.gray, **kwargs)
        plt.axis('off')

    @staticmethod
    def box(b, line_width=2, color='g', style='-'):
        """ Draws a box on the current plot.

        Parameters
        ----------
        b : :obj:`autolab_core.Box`
            box to draw
        line_width : int
            width of lines on side of box
        color : :obj:`str`
            color of box
        style : :obj:`str`
            style of lines to draw
        """
        if not isinstance(b, Box):
            raise ValueError('Input must be of type Box')
            
        # get min pixels
        min_i = b.min_pt[1]
        min_j = b.min_pt[0]
        max_i = b.max_pt[1]
        max_j = b.max_pt[0]
        top_left = np.array([min_i, min_j])
        top_right = np.array([max_i, min_j])
        bottom_left = np.array([min_i, max_j])
        bottom_right = np.array([max_i, max_j])

        # create lines
        left = np.c_[top_left, bottom_left].T
        right = np.c_[top_right, bottom_right].T
        top = np.c_[top_left, top_right].T
        bottom = np.c_[bottom_left, bottom_right].T

        # plot lines
        plt.plot(left[:,0], left[:,1], linewidth=line_width, color=color, linestyle=style)
        plt.plot(right[:,0], right[:,1], linewidth=line_width, color=color, linestyle=style)
        plt.plot(top[:,0], top[:,1], linewidth=line_width, color=color, linestyle=style)
        plt.plot(bottom[:,0], bottom[:,1], linewidth=line_width, color=color, linestyle=style)

    @staticmethod
    def grasp(grasp, color='r', arrow_len=4, arrow_head_len = 2, arrow_head_width = 3,
              arrow_width = 1, jaw_len=3, jaw_width = 3.0,
              grasp_center_size=7.5, grasp_center_thickness=2.5,
              grasp_center_style='+', grasp_axis_width=1,
              grasp_axis_style='--', line_width=8.0, show_center=True, show_axis=False, scale=1.0):
        """
        Plots a 2D grasp with arrow and jaw style using matplotlib
        
        Parameters
        ----------
        grasp : :obj:`Grasp2D`
            2D grasp to plot
        color : :obj:`str`
            color of plotted grasp
        arrow_len : float
            length of arrow body
        arrow_head_len : float
            length of arrow head
        arrow_head_width : float
            width of arrow head
        arrow_width : float
            width of arrow body
        jaw_len : float
            length of jaw line
        jaw_width : float
            line width of jaw line
        grasp_center_thickness : float
            thickness of grasp center
        grasp_center_style : :obj:`str`
            style of center of grasp
        grasp_axis_width : float
            line width of grasp axis
        grasp_axis_style : :obj:`str`
            style of grasp axis line
        show_center : bool
            whether or not to plot the grasp center
        show_axis : bool
            whether or not to plot the grasp axis
        """
        # plot grasp center
        if show_center:
            plt.plot(grasp.center.x, grasp.center.y, c=color, marker=grasp_center_style, mew=scale*grasp_center_thickness, ms=scale*grasp_center_size)
        
        # compute axis and jaw locations
        axis = grasp.axis
        g1 = grasp.center.data - (float(grasp.width_px) / 2) * axis
        g2 = grasp.center.data + (float(grasp.width_px) / 2) * axis
        g1p = g1 - scale * arrow_len * axis # start location of grasp jaw 1
        g2p = g2 + scale * arrow_len * axis # start location of grasp jaw 2

        # plot grasp axis
        if show_axis:
            plt.plot([g1[0], g2[0]], [g1[1], g2[1]], color=color, linewidth=scale*grasp_axis_width, linestyle=grasp_axis_style)
        
        # direction of jaw line
        jaw_dir = scale * jaw_len * np.array([axis[1], -axis[0]])
        
        # length of arrow
        alpha = scale*(arrow_len - arrow_head_len)
        
        # plot first jaw
        g1_line = np.c_[g1p, g1 - scale*arrow_head_len*axis].T
        plt.arrow(g1p[0], g1p[1], alpha*axis[0], alpha*axis[1], width=scale*arrow_width, head_width=scale*arrow_head_width, head_length=scale*arrow_head_len, fc=color, ec=color)
        jaw_line1 = np.c_[g1 + jaw_dir, g1 - jaw_dir].T

        plt.plot(jaw_line1[:,0], jaw_line1[:,1], linewidth=scale*jaw_width, c=color) 

        # plot second jaw
        g2_line = np.c_[g2p, g2 + scale*arrow_head_len*axis].T
        plt.arrow(g2p[0], g2p[1], -alpha*axis[0], -alpha*axis[1], width=scale*arrow_width, head_width=scale*arrow_head_width, head_length=scale*arrow_head_len, fc=color, ec=color)
        jaw_line2 = np.c_[g2 + jaw_dir, g2 - jaw_dir].T
        plt.plot(jaw_line2[:,0], jaw_line2[:,1], linewidth=scale*jaw_width, c=color) 
