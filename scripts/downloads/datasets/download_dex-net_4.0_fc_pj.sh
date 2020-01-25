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

wget -O data/training/dexnet_4_fc_pj_aa https://berkeley.box.com/shared/static/xhv3preqlada05cz38g6mqutvcw7x2pi
wget -O data/training/dexnet_4_fc_pj_ab https://berkeley.box.com/shared/static/8o501rclohrue80eny2dgkh0ftlg660y
wget -O data/training/dexnet_4_fc_pj_ac https://berkeley.box.com/shared/static/khyvf5vw4im0jg46orkix8a8pdnu2t9o
wget -O data/training/dexnet_4_fc_pj_ad https://berkeley.box.com/shared/static/bq9dibanj2tg3zhj5ntcbkbut71rk7y4
wget -O data/training/dexnet_4_fc_pj_ae https://berkeley.box.com/shared/static/oa46t5oz1srqocncxvywpqizwmz5f6by
wget -O data/training/dexnet_4_fc_pj_af https://berkeley.box.com/shared/static/t27a1x89es2g4l4jlm2j8c3brypixb76
wget -O data/training/dexnet_4_fc_pj_ag https://berkeley.box.com/shared/static/09o1gjaqz0s7ol1kmqj1vnyrnynlgozn
wget -O data/training/dexnet_4_fc_pj_ah https://berkeley.box.com/shared/static/s6s9dtl8r6cr3evy7gt13g66dpblowhd
wget -O data/training/dexnet_4_fc_pj_ai https://berkeley.box.com/shared/static/q61i5muzddrmo37899nmyptpme4u42hx
wget -O data/training/dexnet_4_fc_pj_aj https://berkeley.box.com/shared/static/s4q2lkh17dsxsz8zr1dttvcqj5veq1ze

cd data/training
cat dexnet_4_fc_pj_a* > dexnet_4_fc_pj.zip
unzip dexnet_4_fc_pj.zip
mv grasps dex-net_4.0_fc_pj
cd ../..
