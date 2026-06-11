"""Shared body, observation, and genome dimension constants.

Extracted from the integrated reference engine; behavior-preserving move.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Callable, Deque, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, lax, random, vmap

N_JOINTS      = 12   # default / minimum meaningful

TACTILE_ZONES = 12

INTERO_DIM    = 8

MOTOR_DIM     = 6

MAX_JOINTS    = 24

MAX_MOTORS    = 8

MAX_TACTILE   = 24

BODY_VEC_LEN  = 3 + 3 + 4 + 3 + MAX_JOINTS + MAX_JOINTS + MAX_TACTILE + INTERO_DIM + MAX_MOTORS

ATTN_DIM      = 32

FIELD_PATCH_DIM = 64

FIELD_OBS_START = BODY_VEC_LEN + 4 + ATTN_DIM

FIELD_Q_IDX     = FIELD_OBS_START + FIELD_PATCH_DIM

FIELD_GRAD_IDX  = FIELD_Q_IDX + 1

FIELD_STAB_IDX  = FIELD_Q_IDX + 4

FIELD_OBS_DIM   = FIELD_PATCH_DIM + 5

sigmoid = jax.nn.sigmoid

GENOME_DIM = 256

GENOME_MAX_MODULE_DIM = 512   # hard cap per module — prevents unbounded growth

GENOME_DUP_PROB       = 0.02  # probability of segment duplication per module per birth

GENOME_DEL_PROB       = 0.01  # probability of segment deletion per module per birth

GENOME_DUP_SEGMENT    = 8     # elements copied / removed per event

GENOME_LOCI_PER_MODULE  = 4      # sigma-field positions encoding each genome module

GENOME_FIELD_MAINT_COST = 3e-4   # energy / locus / step (metabolic cost of heredity)

GENOME_FIELD_STRENGTH   = 0.35   # pump magnitude when writing genome to field

GENOME_FIELD_RADIUS     = 2      # spatial radius of genome loci around body centre
