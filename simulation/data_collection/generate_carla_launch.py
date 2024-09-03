
source_root = "$CARLA_ROOT"
# source_root = "/home/zu/data/hjh/V2Xverse/external_paths/carla_root"
for i in range(15):
    print(f'DISPLAY=localhost:10.{int(i/4)} CUDA_VISIBLE_DEVICES={i} {source_root}/CarlaUE4.sh --world-port={40000+2*i} -prefer-nvidia')
