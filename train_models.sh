#!/bin/bash
source .venv/bin/activate

gnome-terminal -- bash -c "python3 num_model_v2.py; exec bash"
gnome-terminal -- bash -c "python3 num_model_v3.py; exec bash"
gnome-terminal -- bash -c "python3 num_model_v1.py; exec bash"
gnome-terminal -- bash -c "python3 sym_model_v1.py; exec bash"
gnome-terminal -- bash -c "python3 sym_model_v2.py; exec bash"
gnome-terminal -- bash -c "python3 sym_model_v3.py; exec bash"
gnome-terminal -- bash -c "python3 num_sym_model_v1.py; exec bash"
gnome-terminal -- bash -c "python3 num_sym_model_v1.py; exec bash"
gnome-terminal -- bash -c "python3 num_sym_model_v1.py; exec bash"

deactivate