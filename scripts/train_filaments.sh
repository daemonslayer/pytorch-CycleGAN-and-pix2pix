set -ex
python3 train.py --dataroot ./datasets/filaments --name filaments_pix2pix --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode filaments --norm batch --pool_size 0
