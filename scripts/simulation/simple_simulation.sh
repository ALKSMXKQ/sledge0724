#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

CHALLENGE=sledge_reactive_agents
SCENARIO_CACHE_PATH=/home16T/home8T_1/leitingting/sledge_workspace/sledge/output/lane_regularizer_exact_ab/sim_cache_on

python $SLEDGE_DEVKIT_ROOT/sledge/script/evaluation/run_simulation.py \
  +simulation=$CHALLENGE \
  planner=pdm_closed_planner \
  observation=sledge_agents_observation \
  scenario_builder=nuplan \
  cache.scenario_cache_path=$SCENARIO_CACHE_PATH \
  worker=sequential \
  number_of_cpus_allocated_per_simulation=1 \
  number_of_gpus_allocated_per_simulation=1