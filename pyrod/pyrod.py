""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

Released under the GNU Public Licence v2.

This is the main script to run PyRod from the command line for analyzing molecular dynamics simulations and generating
dMIFs, pharmacophores and centroids.
"""

# Standard library
from typing import Tuple, Sequence, Optional
import argparse
import configparser
import multiprocessing
import os
import time
import warnings
import pickle
from collections import OrderedDict


# 3rd party
import numpy as np
import MDAnalysis as mda

# Self
import pyrod


def chunks(iterable, chunk_size):
    """This functions returns a list of chunks."""
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]


def write_test_grid_to_pdb(
        center: Tuple[float, float, float],
        size: Tuple[float, float, float],
        name: str,
        output_path,
):
    """
    Section for determining center and edge lengths parameters (integers)
    for later dmif analysis. A pdb file will be written containing grid
    points represented as pseudo atoms that can be used to check proper
    grid placement in e.g. pymol or vmd. The center and edge lengths
    parameters define the center and edge lengths of the grid for x, y and
    z axis.
    """
    # determine space resulting in less than 100000 grid points
    space = 0.5
    space_found = False
    while not space_found:
        grid = pyrod.grid.generate_grid(center, size, space)
        if len(grid) < 100000:
            space_found = True
            pyrod.write.pdb_grid(grid, name, "{}/test".format(output_path))
        else:
            if space == 0.5:
                space += 0.5
            else:
                space += 1
    return


def point_properties(
        point: Tuple[float, float, float] = (0, 0, 0),
        dmif_path: str = "/path/to/pyrod/data/dmif.pkl",
):
    """
    Section for defining parameters to retrieve dMIF properties of a grid
    point. The grid point is described by coordinates, e.g. from a
    pharmacophore feature. Path to dmif.pkl needs to be specified.
    """
    with open(dmif_path, "rb") as file:
        dmif = pickle.load(file)
    point_properties = dmif[
        (dmif["x"] == point[0]) & (dmif["y"] == point[1]) & (dmif["z"] == point[2])
    ]
    point_properties_dict = OrderedDict()
    for point_property in point_properties.dtype.names:
        point_properties_dict[point_property] = round(point_properties[point_property][0], 2)
    return point_properties_dict


def trajectory_analysis(
        center: Tuple[int, int, int] = (0,0,0),
        edge_lengths: Tuple[int, int, int] = (30, 30, 30),
        topology: str = "/path/to/topology.pdb",
        trajectories: Sequence[str] = (
                "/path/to/trajectory0.dcd", "/path/to/trajectory1.dcd",
                "/path/to/trajectory2.dcd", "/path/to/trajectory3.dcd", "/path/to/trajectory4.dcd"
        ),
        first_frame: Optional[int] = None,
        last_frame: Optional[int] = None,
        step_size: Optional[int] = None,
        metal_names: Sequence[str] = ["FE"],
        map_formats: Sequence[str] = ("cns", "xplor", "kont"),
        number_of_processes: int = 1,
        dmifs_only: bool = False,
):
    """
    Section for defining parameters for trajectory analysis. The
    parameters for center and edge lengths (in Angstrom) of the grid can be
    determined by using test_grid.cfg. First frame, last frame and step
    size parameters can be used to specify the part of the trajectories to
    analyze. If all three parameters are empty, trajectories will be
    analyzed from beginning to end. Important metals can be specified
    comma-separated as named in the topology file (e.g. FE). By default all
    available map formats are written. Multi processing is supported. If
    pharmacophores are not of interest, the dmifs only parameter can be set
    true, interaction partners will not be recorded resulting in improved
    computational performance and dramatically lower memory usage.
    """
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
    get_partners = not dmifs_only
    grid_score, grid_partners = pyrod.grid.dmif_data_structure(
        pyrod.grid.generate_grid(center, edge_lengths), get_partners
    )
    manager = multiprocessing.Manager()
    results = manager.list()
    frame_counter = multiprocessing.Value("i", 0)
    trajectory_time = time.time()
    processes = [
        multiprocessing.Process(
            target=pyrod.trajectory.trajectory_analysis,
            args=(
                topology,
                trajectory,
                grid_score,
                grid_partners,
                frame_counter,
                total_number_of_frames,
                first_frame,
                last_frame,
                step_size,
                metal_names,
                counter,
                directory,
                debugging,
                get_partners,
                trajectory_time,
                results,
            ),
        )
        for counter, trajectory in enumerate(trajectories)
    ]
    if len(trajectories) > 1:
        pyrod.write.update_user(
            "Analyzing {} frames from {} trajectories.".format(
                total_number_of_frames, len(trajectories)
            ),
            logger,
        )
    else:
        pyrod.write.update_user(
            "Analyzing {} frames from 1 trajectory.".format(total_number_of_frames), logger
        )
    for chunk in chunks(processes, number_of_processes):
        for process in chunk:
            process.start()
        for process in chunk:
            process.join()
    pyrod.write.update_user("Processing results.", logger)
    # convert multiprocessing list to true python list
    results_list = []
    for x in results:
        results_list.append(x)
    results = None
    dmif, partners = pyrod.grid.post_processing(results_list, total_number_of_frames)
    results_list = None
    pyrod.write.update_user("Writing raw data to {}/data.".format(directory), logger)
    pyrod.write.pickle_writer(dmif, "dmif", "{}/{}".format(directory, "data"))
    if get_partners:
        for key in pyrod.lookup.grid_list_dict.keys():
            pyrod.write.pickle_writer(
                partners[key].tolist(), key, "/".join([directory, "data"])
            )
    partners = None
    pyrod.write.update_user("Writing maps to {}/dmifs.".format(directory), logger)
    for map_format in map_formats:
        for feature_type in [x for x in dmif.dtype.names if x not in ["x", "y", "z"]]:
            pyrod.write.dmif_writer(
                dmif[feature_type],
                np.array([[x, y, z] for x, y, z in zip(dmif["x"], dmif["y"], dmif["z"])]),
                map_format,
                feature_type,
                "{}/{}".format(directory, "dmifs"),
                logger,
            )
    return


def exclusion_volume_params(
        dmif: str = "/path/to/pyrod/data/dmif.pkl",
        shape_cutoff: float = 1,
        restrictive: bool = False,
):
    """
    Section for defining parameters for exclusion volume generation based
    on dmifs. Path to dmif.pkl needs to be specified. All grid points
    smaller than shape cutoff will be considered for exclusion volume
    generation. The restrictive parameter can be set true to generate a
    more dense exclusion volume coat.
    """
    dmif = pyrod.read.pickle_reader(dmif, "dmif", logger)
    evs = pyrod.pharmacophore.generate_exclusion_volumes(
        dmif, directory, debugging, shape_cutoff, restrictive
    )
    pyrod.write.pickle_writer(evs, "exclusion_volumes", "/".join([directory, "data"]))
    return


def generate_features(
    dmif: str = "/path/to/pyrod/data/dmif.pkl",
    partners: str = "/path/to/pyrod/data",
    number_of_processes: int = 1,
    features_per_feature_type: int = 20,
):
    """
    Section for defining parameters for feature generation based on
    dmifs. Path to dmif.pkl and the path to the directory containing the
    partner.pkl-files (e.g. ha.pkl etc., usually same directory as for
    dmif.pkl) need to be specified. Specify the number of features to be
    generated per feature type. Multi processing is supported.
    """
    dmif = pyrod.read.pickle_reader(dmif, "dmif", logger)
    partner_path = config.get("feature parameters", "partners")
    features_per_feature_type, number_of_processes = pyrod.config.feature_parameters(config)
    positions = np.array([[x, y, z] for x, y, z in zip(dmif["x"], dmif["y"], dmif["z"])])
    manager = multiprocessing.Manager()
    results = manager.list()
    feature_counter = multiprocessing.Value("i", 0)
    feature_time = time.time()
    processes = [
        multiprocessing.Process(
            target=pyrod.pharmacophore.generate_features,
            args=(
                positions,
                np.array(dmif[feature_type]),
                feature_type,
                features_per_feature_type,
                directory,
                partner_path,
                debugging,
                len(pyrod.lookup.feature_types) * features_per_feature_type,
                feature_time,
                feature_counter,
                results,
            ),
        )
        for feature_type in pyrod.lookup.feature_types
    ]
    pyrod.write.update_user("Generating features.", logger)
    for chunk in chunks(processes, number_of_processes):
        for process in chunk:
            process.start()
        for process in chunk:
            process.join()
    pyrod.write.update_user("Generated {} features.".format(len(results)), logger)
    features = pyrod.pharmacophore_helper.renumber_features(results)
    pyrod.write.pickle_writer(features, "features", "/".join([directory, "data"]))
    return


def generate_pharmacophore(
    exclusion_volumes: str = "/path/to/pyrod/data/exclusion_volumes.pkl",
    features: str = "/path/to/pyrod/data/features.pkl",
    pharmacophore_formats: Sequence[str] = ("pdb", "pml"),
):
    """
    Section for defining parameters for pharmacophore generation. Paths
    to exclusion_volumes.pkl and features.pkl need to be specified.
    Currently, pml (LigandScout) and pdb like pharmacophores can be saved.
    """
    features = pyrod.read.pickle_reader(features, "features", logger)
    evs = pyrod.read.pickle_reader(exclusion_volumes, "exclusion volumes", logger)
    pharmacophore = pyrod.pharmacophore_helper.renumber_features(features + evs)
    evs = [[counter + len(features) + 1] + x[1:] for counter, x in enumerate(evs)]
    pharmacophore_directory = "/".join([directory, "pharmacophores"])
    pyrod.write.update_user(
        "Writing pharmacophore with all features to {}.".format(pharmacophore_directory),
        logger,
    )
    pyrod.write.pharmacophore_writer(
        pharmacophore,
        pharmacophore_formats,
        "super_pharmacophore",
        pharmacophore_directory,
        logger,
    )
    return


def generate_library(
        pharmacophore_path: str = "/path/to/pharmacophore.file",
        output_format: str = "pml",
        minimal_features: int = 3,
        maximal_features: int = 5,
        minimal_hydrogen_bonds: int = 1,
        maximal_hydrogen_bonds: int = 4,
        minimal_hydrophobic_interactions: int = 1,
        maximal_hydrophobic_interactions: int = 3,
        minimal_aromatic_interactions: int = 0,
        maximal_aromatic_interactions: int = 2,
        minimal_ionizable_interactions: int = 0,
        maximal_ionizable_interactions: int = 1,
        library_name: str = "library",
        pyrod_pharmacophore: bool = True
):
    """
    Section for defining parameters for pharmacophore library generation
    based on an input pharmacophore. Currently only pml and pdb-like
    pharmacophores (from pyrod) are supported for in- and output. Mandatory
    features will be present in each pharmacophore. Optional features will
    be added in a combinatorial fashion based on specified parameters
    below. If the pharmacophore was not generated by pyrod set pyrod
    pharmacophore false.
    """
    library_path = "{}/pharmacophores/{}".format(directory, library_name)

    library_dict = {}
    for parameter in [
        "minimal features",
        "maximal features",
        "minimal hydrogen bonds",
        "maximal hydrogen bonds",
        "minimal hydrophobic interactions",
        "maximal hydrophobic interactions",
        "minimal aromatic interactions",
        "maximal aromatic interactions",
        "minimal ionizable interactions",
        "maximal ionizable interactions",
    ]:
        library_dict[parameter] = int(config.get("library parameters", parameter))

    pyrod.pharmacophore.generate_library(
        pharmacophore_path, output_format, library_dict, library_path, pyrod_pharmacophore, directory, debugging
    )
    return


def dmif_excess_parameters(
    dmif1_path: str = "/path/to/pyrod/data/dmif1.pkl",
    dmif2_path: str = "/path/to/pyrod/data/dmif2.pkl",
    dmif1_name: str = "mif1",
    dmif2_name: str = "mif2",
    map_formats: Sequence[str] = ("ns", "xplor", "kont"),
):
    """
    Section for defining parameters for comparing dmifs via dmif excess
    maps. The grid parameters used for both dmif generations need to be the
    same.
    """
    dmif1_excess, dmif2_excess = pyrod.grid.generate_dmif_excess(dmif1_path, dmif2_path)
    pyrod.write.update_user(
        "Writing dmif excess maps to {0}/{1}_excess and {0}/{2}_excess.".format(
            directory, dmif1_name, dmif2_name
        ),
        logger,
    )
    for map_format in map_formats:
        for feature_type in [x for x in dmif1_excess.dtype.names if x not in ["x", "y", "z"]]:
            pyrod.write.dmif_writer(
                dmif1_excess[feature_type],
                np.array([[x, y, z] for x, y, z in zip(dmif["x"], dmif["y"], dmif["z"])]),
                map_format,
                feature_type,
                "{}/{}_excess".format(directory, dmif1_name),
                logger,
            )
            pyrod.write.dmif_writer(
                dmif2_excess[feature_type],
                np.array([[x, y, z] for x, y, z in zip(dmif["x"], dmif["y"], dmif["z"])]),
                map_format,
                feature_type,
                "{}/{}_excess".format(directory, dmif2_name),
                logger,
            )
    return


def centroid_params(
        ligand: str,
        pharmacophore_path: str = "/path/to/pharmacophore.file",
        topology: str = "/path/to/topology.pdb",
        trajectories: Sequence[str] = (
                "/path/to/trajectory0.dcd",
                "/path/to/trajectory1.dcd",
                "/path/to/trajectory2.dcd",
                "/path/to/trajectory3.dcd",
                "/path/to/trajectory4.dcd"
        ),
        first_frame: Optional[int] = None,
        last_frame: Optional[int] = None,
        step_size: Optional[int] = None,
        metal_names: Sequence[str] = ("FE",),
        output_name: str = "centroid",
        number_of_processes: int = 1,
):
    """
    Section for defining parameters for protein centroid generation. The 
    provided trajectories will be screened for protein conformations that  
    match the given pharmacophore and are not involved in severe atom      
    clashes with the ligand (mol2 format, optional). Exclusion volumes are 
    ignored. Matching protein conformations will be analyzed to select a   
    protein centroid. First frame, last frame and step size parameters can 
    be used to specify the part of the trajectories to analyze. If all     
    three parameters are empty, trajectories will be analyzed from         
    beginning to end. Important metals can be specified comma-separated as 
    named in the topology file (e.g. FE). Multi processing is supported. If
    a ligand is not provided, protein conformations are only checked to    
    match the given pharmacophore. Several parameters in the pharmacophore 
    file !have! to be adjusted to allow screening of protein conformations 
    (see box below centroid parameter section). This component will output 
    a csv-file containing information about matching frames, a trajectory  
    of the protein with all frames fitting the pharmacophore (protein.pdb, 
    ensemble.dcd) and the centroid in pdb-format including all atoms, e.g. 
    water, lipids and ions.
    
      The parameters in the pharmacophore file !have! to be modified to     
    provide score minima for hydrophobic, aromatic and ionizable            
    interactions and !can! be further modified to make the search less or   
    more restrictive by altering the tolerance parameters. This can be done 
    directly in the pml or pdb pharmacophore file using a text editor.      
                                                                            
      Score minima (default = 1) are read from the weight parameters in the 
    pml file or the b-factor column in the pdb pharmacophore file and       
    specify the minimal score to accept an interaction in the protein       
    conformation screen:                                                    
      The score of hydrophobic features corresponds to the number of        
    hydrophobic atoms nearby scaled by buriedness and can be estimated with 
    the hi_norm dmif, e.g. a score of 3 will filter protein conformations   
    to allow a hydrophobic score not less than 3.                           
      The score of ionizable features corresponds to the distance between   
    the charged centers (score = 2.6 / distance). Hence, a weight of 1 will 
    filter protein conformations to allow an ionizable interaction with a   
    distance of not more than 2.6 A. Since, ionizable interactions can      
    include multiple charged centers of the protein, a score of e.g. 1.6    
    will most likely filter protein conformations that allow ionizable      
    interactions with 2 or more charged centers of the protein.             
      The score of aromatic features corresponds to the distance between    
    the atom groups. An optimal distance leads to a score of 1 and is       
    different for pi-stacking (3.5 A), t-stacking (5.0 A) and cation-pi     
    interactions (3.8 A). The complete distance based scoring can be found  
    in pyrod/pyrod_lib/lookup.py. Similar to ionizable interactions,        
    aromatic interaction can involve multiple partners leading to a score   
    higher than 1.                                                          
      The scores of hydrogen bonds are ignored.                             
                                                                            
      The restrictiveness of the protein conformation search can be further 
    altered by changing the tolerance parameter of the feature core and the 
    feature partner.                                                        
      The tolerance radius of the feature core (default = 1.5 A) is used as 
    a threshold for clashes with the protein heavy atoms. Hence, a core     
    tolerance radius of 2.5 A of a certain feature will filter protein      
    conformations that lack heavy atoms within 2.5 A of that feature. A     
    minimal core tolerance radius of 2.5 A (corresponding to a relatively   
    short hydrogen bond) is recommended and should in general not be higher 
    for hydrogen bonds and ionizable features without any good reason. The  
    core tolerance radii of hydrophobic and aromatic features can be        
    higher.                                                                 
      The tolerance radius of the feature partner of hydrogen bonds         
    (default = 1.9499999 A) and the tolerance angle for aromatic            
    interactions (default = 0.43633232 rad or 25 °) are used to check for   
    feature partners on the protein site. Decreasing these parameters will  
    make the search more restrictive.                                                                                   
    """
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
    frame_counter = multiprocessing.Value("i", 0)
    trajectory_time = time.time()
    processes = [
        multiprocessing.Process(
            target=pyrod.trajectory.screen_protein_conformations,
            args=(
                topology,
                trajectory,
                pharmacophore_path,
                ligand,
                counter,
                first_frame,
                last_frame,
                step_size,
                metal_names,
                directory,
                output_name,
                debugging,
                total_number_of_frames,
                frame_counter,
                trajectory_time,
            ),
        )
        for counter, trajectory in enumerate(trajectories)
    ]
    if len(trajectories) > 1:
        pyrod.write.update_user(
            "Analyzing {} frames from {} trajectories.".format(
                total_number_of_frames, len(trajectories)
            ),
            logger,
        )
    else:
        pyrod.write.update_user(
            "Analyzing {} frames from 1 trajectory.".format(total_number_of_frames), logger
        )
    for chunk in chunks(processes, number_of_processes):
        for process in chunk:
            process.start()
        for process in chunk:
            process.join()
    pyrod.write.update_user("Finding centroid generation.", logger)
    if debugging:
        pyrod.trajectory.ensemble_to_centroid(
            topology, trajectories, output_name, directory, debugging
        )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pyrod.trajectory.ensemble_to_centroid(
                topology, trajectories, output_name, directory, debugging
            )
    pyrod.write.update_user(
        "Output written to {0}.".format("/".join([directory, output_name])), logger
    )
    return
