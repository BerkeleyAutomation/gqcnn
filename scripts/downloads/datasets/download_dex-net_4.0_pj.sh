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

wget -O data/training/dexnet_4_pj_aa https://berkeley.box.com/shared/static/vx59bt10e7nl409e7oy081ymwdx13sun.0_pj_aa
wget -O data/training/dexnet_4_pj_ab https://berkeley.box.com/shared/static/dujezomcb9228uht952qiek30heo2kvt.0_pj_ab
wget -O data/training/dexnet_4_pj_ac https://berkeley.box.com/shared/static/gzz6jhilvg927ke3ad373rmhpzi8hh60.0_pj_ac
wget -O data/training/dexnet_4_pj_ad https://berkeley.box.com/shared/static/kgnmwexu82t0q5e72zd5vitjylbbu9f7.0_pj_ad
wget -O data/training/dexnet_4_pj_ae https://berkeley.box.com/shared/static/jmiemqczh8wajbo11v94408gz4f3utw4.0_pj_ae
wget -O data/training/dexnet_4_pj_af https://berkeley.box.com/shared/static/b8wi2grdsmr3nulx6l2c8yhd4rda88ul.0_pj_af

cd data/training
cat dexnet_4_pj_a* > dexnet_4_pj.zip
unzip dexnet_4_pj.zip
mv parallel_jaw dexnet_4_pj
cd ../..
