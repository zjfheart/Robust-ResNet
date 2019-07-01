# Training ResNet
python3 run.py --GPU_ID=0  --h=0.1 start_LR=0.1 --n=18 --epochs=80 --seed=7
python3 run.py --GPU_ID=0  --h=1.0 start_LR=0.1 --n=18 --epochs=80 --seed=7

# results visualization 
python3 vis_acc.py 

echo "ALL DONE!"
