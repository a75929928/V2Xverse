for i in range(15):
    print(f"CUDA_VISIBLE_DEVICES={i} $CARLA_ROOT/CarlaUE4.sh --world-port={40000+2*i} -prefer-nvidia")
    # print(f"CUDA_VISIBLE_DEVICES={i} ./external_paths/carla_root/CarlaUE4.sh --world-port={40000+2*i} -prefer-nvidia")

''' OUTPUT
CUDA_VISIBLE_DEVICES=0 $CARLA_ROOT/CarlaUE4.sh --world-port=40000 -prefer-nvidia
CUDA_VISIBLE_DEVICES=1 $CARLA_ROOT/CarlaUE4.sh --world-port=40002 -prefer-nvidia
CUDA_VISIBLE_DEVICES=2 $CARLA_ROOT/CarlaUE4.sh --world-port=40004 -prefer-nvidia
CUDA_VISIBLE_DEVICES=3 $CARLA_ROOT/CarlaUE4.sh --world-port=40006 -prefer-nvidia
CUDA_VISIBLE_DEVICES=4 $CARLA_ROOT/CarlaUE4.sh --world-port=40008 -prefer-nvidia
CUDA_VISIBLE_DEVICES=5 $CARLA_ROOT/CarlaUE4.sh --world-port=40010 -prefer-nvidia
CUDA_VISIBLE_DEVICES=6 $CARLA_ROOT/CarlaUE4.sh --world-port=40012 -prefer-nvidia
CUDA_VISIBLE_DEVICES=7 $CARLA_ROOT/CarlaUE4.sh --world-port=40014 -prefer-nvidia
CUDA_VISIBLE_DEVICES=8 $CARLA_ROOT/CarlaUE4.sh --world-port=40016 -prefer-nvidia
CUDA_VISIBLE_DEVICES=9 $CARLA_ROOT/CarlaUE4.sh --world-port=40018 -prefer-nvidia
CUDA_VISIBLE_DEVICES=10 $CARLA_ROOT/CarlaUE4.sh --world-port=40020 -prefer-nvidia
CUDA_VISIBLE_DEVICES=11 $CARLA_ROOT/CarlaUE4.sh --world-port=40022 -prefer-nvidia
CUDA_VISIBLE_DEVICES=12 $CARLA_ROOT/CarlaUE4.sh --world-port=40024 -prefer-nvidia
CUDA_VISIBLE_DEVICES=13 $CARLA_ROOT/CarlaUE4.sh --world-port=40026 -prefer-nvidia
CUDA_VISIBLE_DEVICES=14 $CARLA_ROOT/CarlaUE4.sh --world-port=40028 -prefer-nvidia
'''