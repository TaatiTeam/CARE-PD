# MotionAGFormer & MoMaSK Evaluation

First remember to put the data as previously mentioned.

## MotionAGFormer:
1. Place checkpoint
    - Copy or download your pretrained or fine-tuned checkpoint into "./checkpoint"

2. Run evaluation
    - python finetune_eval.py --eval-only --checkpoint checkpoint --checkpoint-file [checkpoint-name] \
    --config configs/h36m/MotionAGFormer-small.yaml


## MoMaSK:
1. Place checkpoint
    - Download the checkpoint from Momask Github repo and Copy here: pretext/momask/checkpoints/t2m/text_mot_match/model/finest.tar

2. Update experiment name:
    - change ["name"] in ./checkpoints/t2m/Comp_v6_KLD005/opt.txt according to the checkpoint name

3. Compute dataset statistics:
    - run cd momask & python cal_mean_variance.py

4. Evaluate T2M VQ model:
    - run python eval_t2m_vq.py --gpu_id 0 --name [name] --dataset_name t2m --ext [arbitary-export-name]

