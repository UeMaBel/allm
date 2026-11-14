import os
from allmengine import alu
from allmengine.reference import ReferenceModel

# Trainer workers install their candidate before pytest starts.
if os.environ.get("ALLMENGINE_TRAINER_WORKER") != "1":
    alu.set_model(ReferenceModel())
