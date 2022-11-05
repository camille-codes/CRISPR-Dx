import subprocess


def run_offtargets_default(path_to_offtargets):
    subprocess.run(path_to_offtargets, shell=True)
