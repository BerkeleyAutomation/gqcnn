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

wget -O data/training/dexnet_4_suction_aa https://berkeley.box.com/shared/static/ivg7t1bxgc7m9jufa1adxoadoxxj2os0.0_suction_aa
wget -O data/training/dexnet_4_suction_ab https://berkeley.box.com/shared/static/zl0q7xwd4s6gecbbree6hito7czb7090.0_suction_ab
wget -O data/training/dexnet_4_suction_ac https://berkeley.box.com/shared/static/3ndm5s0l69eo5zdrvlr98smbnmrd2lsm.0_suction_ac
wget -O data/training/dexnet_4_suction_ad https://berkeley.box.com/shared/static/geclz22raqdx6h7e8h5bvvao86rrjph8.0_suction_ad
wget -O data/training/dexnet_4_suction_ae https://berkeley.box.com/shared/static/9b0ruxrkidswwl11icggmndag071geuy.0_suction_ae
wget -O data/training/dexnet_4_suction_af https://berkeley.box.com/shared/static/jwqnedw95k1tgfvmhzti7l0w6spqhduk.0_suction_af
wget -O data/training/dexnet_4_suction_ag https://berkeley.box.com/shared/static/0d7lbx5rdshiox6uii7eqk95ljm0ja3o.0_suction_ag

cd data/training
cat dexnet_4_suction_a* > dexnet_4_suction.zip
unzip dexnet_4_suction.zip
mv suction dexnet_4_suction
cd ../..
