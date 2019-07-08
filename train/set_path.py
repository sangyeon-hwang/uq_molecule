"""\
Module-search-path setter.

By importing this module, you can insert the PARENT directory
of this script to `sys.path`.

This is used by the top-level executable scripts in this directory
so that they can import modules under `../`.
"""
import inspect
import os
import sys

def set_path(parent_index=1):
    """Insert one of the upper directories to `sys.path`.

    Arguments
    ---------
    parent_index: int >= 0
        The number of how much you'd like to go upper:
        0 -> the directory of this module
        1 -> the parent directory
        2 -> the grandparent directory
        ...
    """
    ## The directory of this module.
    here = os.path.dirname(
        os.path.abspath(
            inspect.getfile(
                inspect.currentframe()
    )))

    ## Navigate to upper directories as requested.
    current = here
    for _ in range(parent_index):
        current = os.path.dirname(current)

    ## Set the module search path.
    #+ Inserting to 1 is for leaving '' in the front.
    sys.path.insert(1, current)

set_path()
