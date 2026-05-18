#!/bin/bash

python src/run_2wikiqa.py \
  --agent_config conf.d/agent.yaml \
  --task_config conf.d/2wikiqa.yaml \
  --split dev \
  --mode multi_agent_structured \ 
  --num_examples 100 \
  --output outputs/2wiki_predictions.json

# TODO : mode, num_examples 는 맞춰서 바꿔주시면 됩니다.
# 최종 제출시에는, best performing config으로 맞춰주세요.
# 채점은 bash run_2wikiqa.sh 명령어만으로 수행됩니다.