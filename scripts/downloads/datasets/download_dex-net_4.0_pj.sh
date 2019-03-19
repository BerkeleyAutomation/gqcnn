#!/bin/sh

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

