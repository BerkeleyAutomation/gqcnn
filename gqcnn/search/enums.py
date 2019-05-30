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

Enums for hyper-parameter search.

Author
------
Vishal Satish
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class TrialConstants(object):
    TRIAL_CPU_LOAD = 300  # Decrease to get more aggressize CPU utilization.
    TRIAL_GPU_LOAD = 33  # Decrease to get more aggressize GPU utilization.
    # This really depends on model size (`TRIAL_GPU_LOAD` does too, but it's
    # not a hard limit per se). Ideally we would initialize models one-by-one
    # and monitor the space left, but because model initialization comes after
    # some metric calculation, we set this to be some upper bound based on the
    # largest model and do batch initalizations from there.
    TRIAL_GPU_MEM = 2000


class SearchConstants(object):
    SEARCH_THREAD_SLEEP = 2
    MIN_TIME_BETWEEN_SCHEDULE_ATTEMPTS = 20
