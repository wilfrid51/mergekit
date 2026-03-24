export RAY_record_task_actor_creation_sites=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export TRANSFORMERS_VERBOSITY=info
mergekit-evolve \
  --storage-path ./evolve_storage \
  examples/evolve_config.yml \
  --batch-size 4 \
  --num-gpus 4 \
  --i-understand-the-depths-of-the-evils-i-am-unleashing
  # --task-search-path /root/workspace/mergekit/tasks \
