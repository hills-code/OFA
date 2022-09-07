export MASTER_PORT=8087
user_dir=../../ofa_module
bpe_dir=../../utils/BPE
data=../../dataset/food101/food101_test.tsv
ans2label_file=../../dataset/food101/class2label.pkl
path=food101_checkpoints/5e-5_480/checkpoint_best.pt
result_path=../../results/food101
selected_cols=0,1
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=8087 ../../evaluate.py ../../dataset/food101/food101_test.tsv --path=food101_checkpoints/5e-5_480/checkpoint_best.pt --user-dir=../../ofa_module --task=image_classify --batch-size=8 --log-format=simple --log-interval=10 --seed=7 --gen-subset=val --results-path=../../results/food101 --fp16 --num-workers=0 --model-overrides="{\"data\":\"../../dataset/food101/food101_test.tsv\",\"bpe_dir\":\"../../utils/BPE\",\"selected_cols\":\"0,1\",\"ans2label_file\":\"../../dataset/food101/class2label.pkl\"}"
