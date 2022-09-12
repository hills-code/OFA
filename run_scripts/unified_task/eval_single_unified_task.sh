export MASTER_PORT=8087
user_dir=../../ofa_module
bpe_dir=../../utils/BPE
task_name=sst2
data=../../dataset/unified_data/${task_name}/test.tsv
path=checkpoints/unified_text_cls/${task_name}/10_5e-5_1/checkpoint_last.pt
result_path=../../results/unified_data/${task_name}
selected_cols=0,1
ans2label_dict=../../dataset/unified_data/${task_name}/ans2label.pkl

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=8087 ../../evaluate.py ${data} --path=${path} --user-dir=../../ofa_module --task=unified_text_cls --batch-size=8 --log-format=simple --log-interval=10 --seed=7 --gen-subset=val --results-path=${result_path} --fp16 --num-workers=0 --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"../../utils/BPE\",\"selected_cols\":\"0,1\",\"ans2label_file\":\"${ans2label_dict}\"}"
