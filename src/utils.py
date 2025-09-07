import os


def create_directories(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(os.path.dirname(dir_path), exist_ok=True)
