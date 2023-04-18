from easydict import EasyDict as edict

import numpy as np

# Get the configuration dictionary whose keys can be accessed with dot.
CONFIG = edict()

# ******************************************************************************
# ******************************************************************************
# ******************************************************************************
# GENERAL CONFIGURATIONS
# ******************************************************************************
# ******************************************************************************
# ******************************************************************************
CONFIG.INPUT_JSONS = [
    ('./pointergen_outputv2_17.json', "PointerGen-robert"),
    ('./vanilla6.json', 'Vanilla Seq2seq'),
    # ('./seq2seq49.json', "Seq2Seq + Attn-parnian"), 
    ('./seq2seq_attn_output30.json', "Seq2Seq + Attn-robert"), 
]
# CONFIG.INPUT_JSONS = [ './seq2seq_attn_output30.json']