# EDSR baseline model (x2) + JPEG augmentation
# python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset
#python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset --data_train DIV2K+DIV2K-Q75 --data_test DIV2K+DIV2K-Q75

# EDSR baseline model (x3) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 3 --patch_size 144 --save edsr_baseline_x3 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]

# EDSR baseline model (x4) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 4 --save edsr_baseline_x4 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]

# EDSR in the paper (x2)
#python main.py --model EDSR --scale 2 --save edsr_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset

# EDSR in the paper (x3) - from EDSR (x2)
#python main.py --model EDSR --scale 3 --save edsr_x3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR model dir]

# EDSR in the paper (x4) - from EDSR (x2)
#python main.py --model EDSR --scale 4 --save edsr_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR_x2 model dir]

# MDSR baseline model
#python main.py --template MDSR --model MDSR --scale 2+3+4 --save MDSR_baseline --reset --save_models

# MDSR in the paper
#python main.py --template MDSR --model MDSR --scale 2+3+4 --n_resblocks 80 --save MDSR --reset --save_models

# Standard benchmarks (Ex. EDSR_baseline_x4)
#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --pre_train download --test_only --self_ensemble

#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train download --test_only --self_ensemble

# Test your own images
#python main.py --data_test Demo --scale 4 --pre_train download --test_only --save_results

# Advanced - Test with JPEG images 
#python main.py --model MDSR --data_test Demo --scale 2+3+4 --pre_train download --test_only --save_results

# Advanced - Training with adversarial loss
#python main.py --template GAN --scale 4 --save edsr_gan --reset --patch_size 96 --loss 5*VGG54+0.15*GAN --pre_train download

# RDN BI model (x2)
#python3.6 main.py --scale 2 --save RDN_D16C8G64_BIx2 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 64 --reset
# RDN BI model (x3)
#python3.6 main.py --scale 3 --save RDN_D16C8G64_BIx3 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 96 --reset
# RDN BI model (x4)
#python3.6 main.py --scale 4 --save RDN_D16C8G64_BIx4 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 128 --reset
# python3 main.py --scale 2 --save RDN_D16C8G64_BIx2 --model RDN --epochs 1000 --batch_size 16 --patch_size 64 --save_results --reset --ext sep --GPU_ids 0

# MetaRDN BI model (x4)
# python3 main.py --scale 4 --save MetaRDN_D16C8G64_BIx4 --model MetaRDN --epochs 1000 --batch_size 12 --patch_size 192 --decay 400-600-800-900-1000 --save_results --reset --ext sep --GPU_ids 2
# python3 main.py --save MSMetaRDN_BI --model MetaRDN --epochs 1000 --batch_size 16 --patch_size 50 --save_results --reset --ext sep --GPU_ids 4,5,6,7

# KPRDN BI model (x4)
# python3 main.py --scale 4 --save KPRDN_D16C8G64_BIx4 --model KPRDN --epochs 1000 --batch_size 16 --patch_size 128 --save_results --reset --ext sep --GPU_ids 2

# MyMetaRDN BI model (x4)
python3 main.py --save MyMetaRDN_BI --model MyMetaRDN --epochs 1000 --batch_size 16 --patch_size 50 --save_results --reset --ext sep --GPU_ids 4,5,6,7

# MyMetaRDNV2 BI model (x4)
# python3 main.py --save MyMetaRDNV2_BI --model MyMetaRDNV2 --epochs 1000 --batch_size 16 --patch_size 50 --save_results --reset --ext sep --GPU_ids 0,1,2,3

# KPRDNV1 BI model (x4)
# python3 main.py --save KPRDNV1_BI --model KPRDNV1 --epochs 1000 --batch_size 16 --patch_size 50 --save_results --reset --ext sep --GPU_ids 4,5,6,7

# KPRDNV2 BI model (x4)
# python3 main.py --save MSKPRDNV2_BI --model KPRDNV2 --epochs 1000 --batch_size 16 --patch_size 50 --save_results --reset --ext sep --GPU_ids 4,5,6,7

# KPRDNV3 BI model (x4)
# python3 main.py --save MSKPRDNV3_BI --model KPRDNV3 --epochs 1000 --batch_size 16 --patch_size 50 --save_results --reset --ext sep --GPU_ids 0,1,2,3

# KPRDNV4 BI model (x4)
# python3 main.py --save MSKPRDNV4_BI --model KPRDNV4 --epochs 1000 --batch_size 16 --patch_size 50 --save_results --reset --ext sep --GPU_ids 0,1,2,3

# KPRDNV5 BI model (x4)
# python3 main.py --save MSKPRDNV5_BI --model KPRDNV5 --epochs 1000 --batch_size 16 --patch_size 50 --save_results --reset --ext sep --GPU_ids 4,5,6,7

# KPRDNV6 BI model (x4)
# python3 main.py --save MSKPRDNV6_BI --model KPRDNV6 --epochs 1000 --batch_size 16 --patch_size 50 --save_results --reset --ext sep --GPU_ids 4,5,6,7

# KPRDNV5 BI model (x4)
# python3 main.py --save MSKPRDNV5_BI --model KPRDNV5 --epochs 1000 --batch_size 16 --patch_size 50 --save_results --reset --ext sep --GPU_ids 4,5,6,7

# RCAN_BIX2_G10R20P48, input=48x48, output=96x96
# pretrained model can be downloaded from https://www.dropbox.com/s/mjbcqkd4nwhr6nu/models_ECCV2018RCAN.zip?dl=0
#python main.py --template RCAN --save RCAN_BIX2_G10R20P48 --scale 2 --reset --save_results --patch_size 96
# RCAN_BIX3_G10R20P48, input=48x48, output=144x144
#python main.py --template RCAN --save RCAN_BIX3_G10R20P48 --scale 3 --reset --save_results --patch_size 144 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX4_G10R20P48, input=48x48, output=192x192
#python main.py --template RCAN --save RCAN_BIX4_G10R20P48 --scale 4 --reset --save_results --patch_size 192 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX8_G10R20P48, input=48x48, output=384x384
#python main.py --template RCAN --save RCAN_BIX8_G10R20P48 --scale 8 --reset --save_results --patch_size 384 --pre_train ../experiment/model/RCAN_BIX2.pt
# python3 main.py --template RCAN --save RCAN_BIX2_G10R20P48 --scale 2 --reset --save_results --patch_size 96 --GPU_ids 1 --ext sep

# multiscale kprdn
# python3 main.py --save MSKPRDNV1_BI --model KPRDNV1 --epochs 1000 --batch_size 32 --patch_size 50 --save_results --reset --ext sep --GPU_ids 0,1,2,3 --pre_train ../experiment/MSKPRDNV1_BI_backup/model/model_latest.pt

python3 main.py --save KPRDNV1_BI --model KPRDNV1 --epochs 1000 --batch_size 16 --patch_size 50 --save_models --save_results --reset --ext sep --GPU_ids 4,5,6,7 --pre_train ../experiment/KPRDNV3_BI/model/model_latest.pt 

# test
python3 main.py --save KPEDSR_Test --model KPEDSR --test_only --batch_size 1 --save_results --ext sep --GPU_ids 1 --data_test B100 --pre_train ../experiment/KPEDSR_BI/model/model_latest.pt

python3 main.py --save KPRDN_BI --model KPRDN --epochs 1000 --batch_size 16 --patch_size 50 --save_models --save_results --reset --ext sep --GPU_ids 0,1,2,3

python3 main.py --save KPRDN_BI_Test --model KPRDN --test_only --batch_size 1 --save_results --ext sep --GPU_ids 4 --data_test Set5 --pre_train ../experiment/KPRDN_BI/model/model_latest.pt

python3 main.py --save AdaRDN_BI --model AdaRDN --epochs 1000 --batch_siz16 --patch_size 50 --loss 1*L1+1*Depth --save_models --save_results --reset --ext sep --GPU_ids 0,1,2,3

python3 main.py --save AdaEDSR_BI_Test --model AdaEDSR --test_only --batch_size 1 --save_results --ext sep --GPU_ids 5 --data_test Set5 --pre_train ../experiment/AdaEDSR_BI/model/model_latest.pt

python3 -m torch.distributed.launch main.py --save AdaRDN_BI --model AdaRDN --epochs 1000 --batch_size 16 --patch_size 50 --loss 1*L1+1*Depth --save_models --save_results --reset --ext sep --GPU_ids 0,1,2,3

python37 main.py --model BiRDNV1 --ext img  --save BiRDNV1_BI_Test_DIV2K --GPU_ids 5 --batch_size 1 --test_only --data_test DIV2K --pre_train  ../experiment/BiRDNV1_BI/model/model_latest.pt  --save_results

python37 main.py --model BiRDNV3 --ext sep  --save BiRDNV3_BI_907_Test_Urban100 --GPU_ids 5 --batch_size 1 --test_only --data_test Urban100 --pre_train  ../experiment/BiRDNV3_BI/model/model_907.pt  --save_results


python37 main.py --model BiCARN --ext img --save BiCARN_BI_Test_Urban100 --GPU_ids 0 --batch_size 1 --test_only --data_test Urban100 --pre_train ../experiment/BiCARN_BI/model/model_best.pt --save_results