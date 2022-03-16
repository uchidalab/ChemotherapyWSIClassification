#!/bin/sh

python predict.py --config_path=../config/config_src_10_valwsi_LEV0.yaml --patch_dir=mnt3_LEV0/ --pred_dir=mnt4_LEV0/
python predict.py --config_path=../config/config_src_10_valwsi_LEV1.yaml --patch_dir=mnt3_LEV1/ --pred_dir=mnt4_LEV1/ 