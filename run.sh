# python test.py --ckpt_file wandb/aue8088-pa1/07c4pd0y/checkpoints/epoch\=41-step\=16422.ckpt --model_name MyNetwork_case1
# python test.py --ckpt_file wandb/aue8088-pa1/98xvby95/checkpoints/epoch\=49-step\=19550.ckpt --model_name MyNetwork_case2
# python test.py --ckpt_file wandb/aue8088-pa1/67t64jni/checkpoints/epoch\=35-step\=14076.ckpt --model_name MyNetwork_case3
# python test.py --ckpt_file wandb/aue8088-pa1/hn55xkh9/checkpoints/epoch\=19-step\=15640.ckpt --model_name efficientnet_b0
# python test.py --ckpt_file wandb/aue8088-pa1/b8hdhavp/checkpoints/epoch\=21-step\=17204.ckpt --model_name efficientnet_b1
# python test.py --ckpt_file wandb/aue8088-pa1/zo0qa5dk/checkpoints/epoch\=21-step\=17204.ckpt --model_name efficientnet_b2
# python test.py --ckpt_file wandb/aue8088-pa1/jelous6d/checkpoints/epoch\=23-step\=18768.ckpt --model_name efficientnet_b3
# python test.py --ckpt_file wandb/aue8088-pa1/ys3jdcyr/checkpoints/epoch\=28-step\=22678.ckpt --model_name efficientnet_b4
# python test.py --ckpt_file wandb/aue8088-pa1/fk1l0wep/checkpoints/epoch\=35-step\=28152.ckpt --model_name efficientnet_b5
# python test.py --ckpt_file wandb/aue8088-pa1/jvy157cy/checkpoints/epoch\=14-step\=11730.ckpt --model_name efficientnet_b6
# python test.py --ckpt_file wandb/aue8088-pa1/2d2mxc3b/checkpoints/epoch\=40-step\=32062.ckpt --model_name efficientnet_b7
# python test.py --ckpt_file wandb/aue8088-pa1/8fbxt6l1/checkpoints/epoch\=17-step\=14076.ckpt --model_name resnet18
# python test.py --ckpt_file wandb/aue8088-pa1/34k6t3vu/checkpoints/epoch\=19-step\=15640.ckpt --model_name resnet34
# python test.py --ckpt_file wandb/aue8088-pa1/gvw6lyob/checkpoints/epoch\=21-step\=34386.ckpt --model_name resnet50

python train.py --model_name MyNetwork_case1 --batch_size 256 --lr 0.0003 
python train.py --model_name MyNetwork_case1 --batch_size 256 --milestones 20 30
