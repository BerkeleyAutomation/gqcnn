#!/bin/sh

wget -O data/training/dexnet_4_pj_aa https://berkeley.box.com/shared/static/ybqo02q6471odc2k80pltjwj24dh1gkz.0_pj_aa
wget -O data/training/dexnet_4_pj_ab https://berkeley.box.com/shared/static/3id22ohgprdiv02ne031dgue0oe1r264.0_pj_ab
wget -O data/training/dexnet_4_pj_ac https://berkeley.box.com/shared/static/9p49ilrcgi3t50rst3txo92ocr25ng2u.0_pj_ac
wget -O data/training/dexnet_4_pj_ad https://berkeley.box.com/shared/static/ho8huc2npe0rp9ji3c7wyd1j01cw85lt.0_pj_ad
wget -O data/training/dexnet_4_pj_ae https://berkeley.box.com/shared/static/atsgvq8nxsv7qtidmgb1bwwlv603zrrj.0_pj_ae
wget -O data/training/dexnet_4_pj_af https://berkeley.box.com/shared/static/l373cbiecetmchphiwp7qyxhq3823v66.0_pj_af
wget -O data/training/dexnet_4_pj_ag https://berkeley.box.com/shared/static/8gulaeciyuisa52iyenxdi0c3boa65zn.0_pj_ag

wget -O data/training/dexnet_4_suction_aa https://berkeley.box.com/shared/static/ev8wc4xf6m1zf61wrud18qbr2y4f7wyn.0_suction_aa
wget -O data/training/dexnet_4_suction_ab https://berkeley.box.com/shared/static/1dbkz11fnspxk8bg379lqo0931i4cmxx.0_suction_ab
wget -O data/training/dexnet_4_suction_ac https://berkeley.box.com/shared/static/xmlhcx3tl40jq01wwepsg46pkbtge3sp.0_suction_ac
wget -O data/training/dexnet_4_suction_ad https://berkeley.box.com/shared/static/s1l2jjucn44gwzf8hcmz2lh0ntqlmhy4.0_suction_ad
wget -O data/training/dexnet_4_suction_ae https://berkeley.box.com/shared/static/rx6dnal6yieb01se9kanrqfbnf0iayix.0_suction_ae
wget -O data/training/dexnet_4_suction_af https://berkeley.box.com/shared/static/n4ncoqeqa086ijua28gvc78gg763v1r7.0_suction_af
wget -O data/training/dexnet_4_suction_ag https://berkeley.box.com/shared/static/2jqr1mki23iw4xsq2lykh9kdoh0g0v04.0_suction_ag

cd data/training
cat dexnet_4_pj_a* > dexnet_4_pj.zip
unzip -a dexnet_4_pj.zip
cat dexnet_4_suction_a* > dexnet_4_suction.zip
unzip -a dexnet_4_suction.zip
cd ../..
