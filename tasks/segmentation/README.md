```bash
torchrun --nproc_per_node=2 ./tasks/segmentation/train_distr.py
torchrun --nproc_per_node=2 ./tasks/segmentation/train_multi.py  --model-name DINOv3 --num-modalities 1 --dataset-name Vaihingen
```

# 重新挂载并指定用户和组权限
```bash
sudo mount -t cifs //10.196.21.55/20251105 /home/yyyjvm/SS-datasets/YYYJ_dataset/Desktop-vvgck54/20251105 -o username=dell,password=guojia1995,uid=1000,gid=1000,file_mode=0777,dir_mode=0777
sudo umount  /home/yyyjvm/SS-datasets/YYYJ_dataset/Desktop-vvgck54/20251105
```

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 ./tasks/segmentation/train_multi.py
```