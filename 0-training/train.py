#!/usr/bin/env python
# coding: utf-8

import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import argparse

import equinox as eqx
import jax
import jax.random as jr
import optax 

import feedbax

import rnns_learn_robust_motor_policies
from rnns_learn_robust_motor_policies import PROJECT_SEED

from rnns_learn_robust_motor_policies.database import (
    get_db_session,
)
from rnns_learn_robust_motor_policies.misc import log_version_info
from rnns_learn_robust_motor_policies.training import (
    train_and_save_models,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train some models on some tasks based on a config file.")
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()
    
    version_info = log_version_info(
        jax, eqx, optax, git_modules=(feedbax, rnns_learn_robust_motor_policies),
    )
    
    db_session = get_db_session()
    
    key = jr.PRNGKey(PROJECT_SEED)
    
    model_records = train_and_save_models(
        db_session, 
        args.config_path, 
        key,
    )