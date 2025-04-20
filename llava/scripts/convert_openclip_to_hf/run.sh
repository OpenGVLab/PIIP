set -x
partition='INTERN3'
# partition='VC2'
TYPE='spot'
# TYPE='reserved'
JOB_NAME='flops'
GPUS=1
GPUS_PER_NODE=1
CPUS_PER_TASK=12
# SRUN_ARGS="--jobid=3699841"
# SRUN_ARGS=""

srun -p $partition --job-name=${JOB_NAME} ${SRUN_ARGS} \
  --gres=gpu:${GPUS_PER_NODE} --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} -n${GPUS} \
  --quotatype=${TYPE} --kill-on-bad-exit=1 \
  bash -c " \
echo start \
&& python -u test.py \
"

# & && python -u convert.py \
