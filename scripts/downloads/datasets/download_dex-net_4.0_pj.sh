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

wget -O data/training/dexnet_4_pj_aa https://berkeley.box.com/shared/static/ybqo02q6471odc2k80pltjwj24dh1gkz.0_pj_aa
wget -O data/training/dexnet_4_pj_ab https://berkeley.box.com/shared/static/3id22ohgprdiv02ne031dgue0oe1r264.0_pj_ab
wget -O data/training/dexnet_4_pj_ac https://berkeley.box.com/shared/static/9p49ilrcgi3t50rst3txo92ocr25ng2u.0_pj_ac
wget -O data/training/dexnet_4_pj_ad https://berkeley.box.com/shared/static/ho8huc2npe0rp9ji3c7wyd1j01cw85lt.0_pj_ad
wget -O data/training/dexnet_4_pj_ae https://berkeley.box.com/shared/static/atsgvq8nxsv7qtidmgb1bwwlv603zrrj.0_pj_ae
wget -O data/training/dexnet_4_pj_af https://berkeley.box.com/shared/static/l373cbiecetmchphiwp7qyxhq3823v66.0_pj_af
wget -O data/training/dexnet_4_pj_ag https://berkeley.box.com/shared/static/8gulaeciyuisa52iyenxdi0c3boa65zn.0_pj_ag

cd data/training
cat dexnet_4_pj_a* > dexnet_4_pj.zip
unzip -a dexnet_4_pj.zip
cd ../..

