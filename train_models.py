import subprocess

venv_path = ".venv\\Scripts\\activate"

scripts = [
    "num_model_v2.py",
    "num_model_v3.py",
    "num_model_v1.py",
    "sym_model_v1.py",
    "sym_model_v2.py",
    "sym_model_v3.py",
    "num_sym_model_v1.py",
    "num_sym_model_v1.py",
    "num_sym_model_v1.py"
]

processes = []

for script in scripts:
    command = f'cmd /k "{venv_path} & python {script}"'
    process = subprocess.Popen(command, shell=True)
    processes.append(process)

for process in processes:
    process.wait()
