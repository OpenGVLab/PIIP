#! Directory of finetuned checkpoint name
CHECKPOINT_PATH=/path/to/PIIP-LLaVA_CLIP-BL_512-256_7B

RUNNAME=$(basename "$CHECKPOINT_PATH")


SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
REPO_DIR="$SCRIPT_DIR/../"



echo runname $RUNNAME

mkdir -p ./eval_results/mmbench/
mkdir -p ./eval_results/mm-vet/
mkdir -p ./eval_results/vqav2/
mkdir -p ./eval_results/seed_bench/

echo running TextVQA
cd $REPO_DIR
python -m llava.eval.model_vqa_loader \
    --model-path $CHECKPOINT_PATH \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/textvqa/train_images/ \
    --answers-file ./playground/data/eval/textvqa/answers/$RUNNAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --dtype fp16
python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/$RUNNAME.jsonl 2>&1 | tee $CHECKPOINT_PATH/eval_textvqa.log





echo running SQA
cd $REPO_DIR
python -m llava.eval.model_vqa_science \
    --model-path $CHECKPOINT_PATH \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test/ \
    --answers-file ./playground/data/eval/scienceqa/answers/$RUNNAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --dtype fp16 \
    --single-pred-prompt
python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$RUNNAME.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${RUNNAME}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${RUNNAME}_result.json 2>&1 | tee $CHECKPOINT_PATH/eval_sqa.log


echo running MMBench
cd $REPO_DIR
python -m llava.eval.model_vqa_mmbench \
    --model-path $CHECKPOINT_PATH \
    --question-file ./playground/data/eval/mmbench/mmbench_dev_20230712.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/mmbench_dev_20230712/$RUNNAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1
python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/mmbench_dev_20230712.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/mmbench_dev_20230712 \
    --upload-dir ./eval_results/mmbench \
    --experiment $RUNNAME


echo running MMVet
cd $REPO_DIR
python -m llava.eval.model_vqa \
    --model-path $CHECKPOINT_PATH \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/$RUNNAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1
python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$RUNNAME.jsonl \
    --dst ./eval_results/mm-vet/$RUNNAME.json


echo running GQA
cd $REPO_DIR
python -m llava.eval.model_vqa_loader \
    --model-path $CHECKPOINT_PATH \
    --question-file ./playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl \
    --image-folder ./playground/data/eval/gqa/data/images \
    --answers-file ./playground/data/eval/gqa/answers/$RUNNAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --dtype fp16
python scripts/convert_gqa_for_eval.py --src ./playground/data/eval/gqa/answers/$RUNNAME.jsonl --dst ./playground/data/eval/gqa/data/testdev_balanced_predictions.json
cd $REPO_DIR/playground/data/eval/gqa/data/
python eval/eval.py --tier testdev_balanced 2>&1 | tee $CHECKPOINT_PATH/eval_gqa.log


echo running SEED
cd $REPO_DIR
python -m llava.eval.model_vqa_loader \
    --model-path $CHECKPOINT_PATH \
    --question-file ./playground/data/eval/seed_bench/llava-seed-bench.jsonl \
    --image-folder ./playground/data/eval/seed_bench \
    --answers-file ./playground/data/eval/seed_bench/answers/$RUNNAME/merge.jsonl \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode vicuna_v1
python scripts/convert_seed_for_submission.py \
    --annotation-file ./playground/data/eval/seed_bench/SEED-Bench.json \
    --result-file ./playground/data/eval/seed_bench/answers/$RUNNAME/merge.jsonl \
    --result-upload-file ./eval_results/seed_bench/$RUNNAME.jsonl | tee $CHECKPOINT_PATH/eval_seed.log


echo running POPE
cd $REPO_DIR
python -m llava.eval.model_vqa_loader \
    --model-path $CHECKPOINT_PATH \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/coco/val2014/ \
    --answers-file ./playground/data/eval/pope/answers/$RUNNAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --dtype fp16
python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco/ \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$RUNNAME.jsonl 2>&1 | tee $CHECKPOINT_PATH/eval_pope.log


echo running VQAv2
cd $REPO_DIR
python -m llava.eval.model_vqa_loader \
    --model-path $CHECKPOINT_PATH \
    --question-file ./playground/data/eval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl \
    --image-folder ./playground/data/eval/vqav2/test2015 \
    --answers-file ./playground/data/eval/vqav2/answers/llava_vqav2_mscoco_test-dev2015/$RUNNAME/merge.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1
python scripts/convert_vqav2_for_submission.py --split llava_vqav2_mscoco_test-dev2015 --ckpt $RUNNAME \
    --dir ./playground/data/eval/vqav2/ \
    --dst ./eval_results/vqav2/

cd $REPO_DIR
python scripts/format_eval_results.py --dir $CHECKPOINT_PATH/ | tee $CHECKPOINT_PATH/eval_all.log
