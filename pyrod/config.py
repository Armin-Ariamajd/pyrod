""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

Released under the GNU Public Licence v2.

This module reads parameters from config files.
"""

# python standard libraries
import warnings

# external libraries
import MDAnalysis as mda

# pyrod modules
try:
    from pyrod.pyrod_lib.lookup import feature_types
    from pyrod.pyrod_lib.write import update_user
except ImportError:
    from pyrod.lookup import feature_types
    from pyrod.write import update_user


def dmif_excess_parameters(config):
    dmif1_path = config.get("dmif excess parameters", "dmif 1")
    dmif2_path = config.get("dmif excess parameters", "dmif 2")
    dmif1_name = config.get("dmif excess parameters", "dmif 1 name")
    if len(dmif1_name) == 0:
        dmif1_name = "dmif1"
    dmif2_name = config.get("dmif excess parameters", "dmif 2 name")
    if len(dmif2_name) == 0:
        dmif2_name = "dmif2"
    map_formats = []
    if len(config.get("dmif excess parameters", "map formats")) > 0:
        map_formats = [
            x.strip() for x in config.get("dmif excess parameters", "map formats").split(",")
        ]
    return [dmif1_path, dmif2_path, dmif1_name, dmif2_name, map_formats]


def centroid_parameters(config, debugging):
    ligand = config.get("centroid parameters", "ligand")
    pharmacophore = config.get("centroid parameters", "pharmacophore")
    topology = config.get("centroid parameters", "topology")
    trajectories = [
        x.strip() for x in config.get("centroid parameters", "trajectories").split(",")
    ]
    first_frame = None
    if len(config.get("centroid parameters", "first frame")) > 0:
        first_frame = (
            int(config.get("centroid parameters", "first frame")) - 1
        )  # convert to zero-based
    last_frame = None
    if len(config.get("centroid parameters", "last frame")) > 0:
        last_frame = int(config.get("centroid parameters", "last frame"))
    step_size = None
    if len(config.get("centroid parameters", "step size")) > 0:
        step_size = int(config.get("centroid parameters", "step size"))
    if debugging:
        total_number_of_frames = len(
            mda.Universe(trajectories[0]).trajectory[first_frame:last_frame:step_size]
        ) * len(trajectories)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            total_number_of_frames = len(
                mda.Universe(trajectories[0]).trajectory[first_frame:last_frame:step_size]
            ) * len(trajectories)
    metal_names = [x.strip() for x in config.get("centroid parameters", "metal names").split(",")]
    if len(config.get("centroid parameters", "output name")) > 0:
        output_name = config.get("centroid parameters", "output name")
    else:
        output_name = "centroid"
    number_of_processes = int(config.get("centroid parameters", "number of processes"))
    return [
        ligand,
        pharmacophore,
        topology,
        trajectories,
        first_frame,
        last_frame,
        step_size,
        total_number_of_frames,
        metal_names,
        output_name,
        number_of_processes,
    ]
