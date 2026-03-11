```bash
torchrun --nproc_per_node=2 ./tasks/segmentation/train_distr.py
torchrun --nproc_per_node=2 ./tasks/segmentation/train_multi.py  --model-name DINOv3 --num-modalities 1 --dataset-name Vaihingen
```

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 ./tasks/segmentation/train_multi.py
```