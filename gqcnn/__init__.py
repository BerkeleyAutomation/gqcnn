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
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .model import get_gqcnn_model, get_fc_gqcnn_model
from .training import get_gqcnn_trainer
from .grasping import (RobustGraspingPolicy, UniformRandomGraspingPolicy,
                       CrossEntropyRobustGraspingPolicy, RgbdImageState,
                       FullyConvolutionalGraspingPolicyParallelJaw,
                       FullyConvolutionalGraspingPolicySuction)
from .analysis import GQCNNAnalyzer
from .search import GQCNNSearch
from .utils import NoValidGraspsException, NoAntipodalPairsFoundException

__all__ = [
    "get_gqcnn_model", "get_fc_gqcnn_model", "get_gqcnn_trainer",
    "RobustGraspingPolicy", "UniformRandomGraspingPolicy",
    "CrossEntropyRobustGraspingPolicy", "RgbdImageState",
    "FullyConvolutionalGraspingPolicyParallelJaw",
    "FullyConvolutionalGraspingPolicySuction", "GQCNNAnalyzer", "GQCNNSearch",
    "NoValidGraspsException", "NoAntipodalPairsFoundException"
]
