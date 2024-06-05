cd deformable_attention

partition='INTERN3'
TYPE='spot'
JOB_NAME='test'
GPUS=1
GPUS_PER_NODE=1
CPUS_PER_TASK=12

srun -p $partition --job-name=${JOB_NAME} ${SRUN_ARGS} \
  --gres=gpu:${GPUS_PER_NODE} --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} -n${GPUS} \
  --quotatype=${TYPE} --kill-on-bad-exit=1 \
    python setup.py install
    
cd ../