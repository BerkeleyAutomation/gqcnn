#!/bin/bash

# Copyright ©2017. The Regents of the University of California (Regents).
# All Rights Reserved. Permission to use, copy, modify, and distribute this
# software and its documentation for educational, research, and not-for-profit
# purposes, without fee and without a signed licensing agreement, is hereby
# granted, provided that the above copyright notice, this paragraph and the
# following two paragraphs appear in all copies, modifications, and
# distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150
# Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
# otl@berkeley.edu,
# http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

# IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
# INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
# THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
# HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
# MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

mkdir -p models

# STANDARD.
wget -O models/GQCNN-2.0.zip https://berkeley.box.com/shared/static/j4k4z6077ytucxpo6wk1c5hwj47mmpux.zip
wget -O models/GQCNN-2.1.zip https://berkeley.box.com/shared/static/zr1gohe29r2dtaaq20iz0lqcbk5ub07y.zip
wget -O models/GQCNN-3.0.zip https://berkeley.box.com/shared/static/8l47knzbzffu8zb9y5u46q0g0rvtuk74.zip
wget -O models/GQCNN-4.0-PJ.zip https://berkeley.box.com/shared/static/boe4ilodi50hy5as5zun431s1bs7t97l.zip
wget -O models/GQCNN-4.0-SUCTION.zip https://berkeley.box.com/shared/static/kzg19axnflhwys9t7n6bnuqsn18zj9wy.zip

# FULLY-CONVOLUTIONAL.
wget -O models/FC-GQCNN-4.0-PJ.zip https://berkeley.box.com/shared/static/d9tvdnudd7f0743gxixcn0k0jeg1ds71.zip
wget -O models/FC-GQCNN-4.0-SUCTION.zip https://berkeley.box.com/shared/static/ini7q54957u0cmaaxfihzn1i876m0ghd.zip

cd models
unzip -a GQCNN-2.0.zip
mv GQ-Image-Wise GQCNN-2.0
unzip -a GQCNN-2.1.zip
mv GQ-Bin-Picking-Eps90 GQCNN-2.1
unzip -a GQCNN-3.0.zip
mv GQ-Suction GQCNN-3.0
unzip -a GQCNN-4.0-PJ.zip
unzip -a GQCNN-4.0-SUCTION.zip
unzip -a FC-GQCNN-4.0-PJ.zip
unzip -a FC-GQCNN-4.0-SUCTION.zip
cd ..
