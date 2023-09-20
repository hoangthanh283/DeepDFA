if [ $# -lt 1 ]
then
echo fail
exit 1
fi

seed=$1
data_dir="/mnt/BASILISK/Files/ICLR2023/LineVul/data/subsets/cross_project/fold_${seed}_holdout"

python linevul_main.py \
  --model_name=$1 \
  --output_dir=./saved_models_uxc--no_flowgnn_crossproject/ \
  --model_type=roberta \
  --tokenizer_name=microsoft/unixcoder-base \
  --model_name_or_path=microsoft/unixcoder-base \
  --do_test \
  --test_data_file=$data_dir/holdout.csv \
  --epochs 10 \
  --block_size 512 \
  --train_batch_size 4 --gradient_accumulation_steps 4 \
  --eval_batch_size 4 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed $seed --no_flowgnn 2>&1 | tee "evalu_noflowgnn_uxc_${dataset}_${seed}_crossproject.log"
