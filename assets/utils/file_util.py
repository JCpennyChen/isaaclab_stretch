import os
import shutil
from typing import Dict, Union



import yaml
from yaml import Loader

from curobo.util.logger import log_warn


def join_path(path1: str, path2: str) -> str:
    """Join two paths, considering OS specific path separators.

    Args:
        path1: Path prefix.
        path2: Path suffix. If path2 is an absolute path, path1 is ignored.

    Returns:
        str: Joined path.
    """
    if path1[-1] == os.sep:
        log_warn("path1 has trailing slash, removing it")
    if isinstance(path2, str):
        return os.path.join(os.sep, path1 + os.sep, path2)
    else:
        return path2


def get_filename(file_path: str, remove_extension: bool = False) -> str:
    """Get file name from file path, removing extension if required.

    Args:
        file_path: Path of file.
        remove_extension: If True, remove file extension.

    Returns:
        str: File name.
    """

    _, file_name = os.path.split(file_path)
    if remove_extension:
        file_name = os.path.splitext(file_name)[0]
    return file_name



def get_path_of_dir(file_path: str) -> str:
    """Get path of directory containing the file.

    Args:
        file_path: Path of file.

    Returns:
        str: Path of directory containing the file.
    """
    dir_path, _ = os.path.split(file_path)
    return dir_path


def get_main_folder_path() -> str:
    """Get the main folder path of the project.

    Returns:
        str: Main folder path.
    """
    return os.path.join(get_path_of_dir(__file__), "..", "..")

def get_assets_path_own_folder():
    """
    Get the path to the 'assets' directory.
    
    """

    

    return os.path.join(get_main_folder_path(), "assets")

def get_configs_path_own_folder():
    return os.path.join(get_main_folder_path(), "configs")

def get_robot_configs_path_own_folder():

    """Get the path to the 'robot_configs' configuration directory.

    """

    return os.path.join(get_configs_path_own_folder(), "robot_configs")


def get_world_configs_path_own_folder() -> str:
    """Get path to world configuration directory in cuRobo.

    World configuration directory contains world configuration files in yaml format. World
    information includes obstacles represented with respect to the robot base frame.

    Returns:
        str: path to world configuration directory.
    """
    config_path = get_configs_path_own_folder()
    path = os.path.join(config_path, "world")
    return path


def load_yaml(file_path: Union[str, Dict]) -> Dict:
    """Load yaml file and return as dictionary. If file_path is a dictionary, return as is.

    Args:
        file_path: File path to yaml file or dictionary.

    Returns:
        Dict: Dictionary containing yaml file content.
    """
    if isinstance(file_path, str):
        with open(file_path) as file_p:
            yaml_params = yaml.load(file_p, Loader=Loader)
    else:
        yaml_params = file_path
    return yaml_params
