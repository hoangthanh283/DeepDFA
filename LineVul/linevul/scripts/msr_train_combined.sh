if [ $# -lt 2 ]
then
echo fail
exit 1
fi

seed=$1
dataset=$2
shift
shift

CUDA_VISIBLE_DEVICES=0 python linevul_main.py \
  --model_name=${seed}_linevul.bin \
  --output_dir=./saved_models \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_train \
  --do_eval \
  --do_test \
  --train_data_file=../data/$dataset/train.csv \
  --eval_data_file=../data/$dataset/val.csv \
  --test_data_file=../data/$dataset/test.csv \
  --epochs 4 \
  --block_size 512 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --eval_batch_size 3 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed $seed $@ 2>&1 | tee "train_${dataset}_${seed}.log"
