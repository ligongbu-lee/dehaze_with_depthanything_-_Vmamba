python ./train.py --opt "ir-sde_train_gan.yml" --load_name "" --multi_gpu "true"  --save_path "./models/models_rain100H" --sample_path "./samples" save_mode 'epoch' --save_by_epoch 250 --save_by_iter 10000 --lr_g 0.0002 --b1 0.5 --b2 0.999 --weight_decay 0.0 --train_batch_size 16 --epochs 5000 --lr_decrease_epoch 2000 --num_workers 1 --crop_size 256 --no_gpu "false" --rainaug "false" 

dataroot 在 ir-sde中进行设置
python ./train.py --save_path "./models/models_realsr" --sample_path "./samples" --weight_decay 0.0 --train_batch_size 4  --epochs 5000 --lr_decrease_epoch 2000


nohup python ./train.py --save_path "./models/models_realsr" --sample_path "./samples" --weight_decay 0.0 --train_batch_size 4  --epochs 5000 --lr_decrease_epoch 2000 >nohup1.out 2>&1 &



python ./validation.py --load_name "./models/models_realsr/KPN_rainy_image_epoch210_bs4.pth"  --save_name "./results/result_sr" --baseroot "./datasets/testing" 


python ./train.py --save_path "./models/models_mixture" --sample_path "./samples" --weight_decay 0.0 --train_batch_size 8  --epochs 5000 --lr_decrease_epoch 2000


python ./validation.py --load_name "./models/models_realsr_medical/KPN_rainy_image_epoch2850_bs4.pth"  --save_name "./results/result_sr" --baseroot ".datasets/LQ_256_mu/train/bengin/" 
