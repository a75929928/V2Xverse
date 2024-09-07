hypes_yaml=opencood/hypes_yaml/v2xverse/where2comm_codebook.yaml
# save_path=opencood/logs/feature_folder
seg_num=2
dict_size=128

cd /home/hjh/carla/Gym/V2Xverse
source /home/hjh/miniconda3/bin/activate v2xverse

python /home/hjh/carla/Gym/V2Xverse/opencood/tools/codebook_trainer.py -y $hypes_yaml --save_path $save_path # --seg_num $seg_num --dict_size $dict_size   
# python /home/hjh/carla/Gym/V2Xverse/opencood/tools/codebook_trainer.py -y $hypes_yaml --model_dir $model_dir 