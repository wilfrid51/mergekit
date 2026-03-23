export RAY_record_task_actor_creation_sites=true
mergekit-evolve \
  --storage-path ./evolve_storage \
  examples/evolve_config.yml \
  --batch-size 4 \
  --task-search-path /root/workspace/mergekit/tasks \
  --num-gpus 4 \
  --i-understand-the-depths-of-the-evils-i-am-unleashing
