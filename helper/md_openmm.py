import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as unit
from sys import stdout
import os

pdb_file = 'name'  # name of pdb file without file ending for mds
directory = '/path/to/directory'  # location of pdb file
passed_mds = 0  # number of mds that have been already generated
number_mds = 10  # number of mds that should be generated
md_length = 10  # length of md in nano seconds
main_forcefield = 'amber14-all.xml'
water_model = 'amber14/tip4pew.xml'

# restrain protein heavy atoms
restrain_heavy_atoms = False
non_protein_resnames = ['HOH', 'NA', 'CL']  # resnames of atoms that should not be restrained

# custom ids of atoms to restrain (check in pdb file, but id of 206 in pdb is 205 in openmm)
restrain_ids = []

# internals
water_model_solvate = 'amber14/tip3p.xml'
directory = directory + '/'

# create solvated system
if passed_mds == 0:
    pdb = app.PDBFile(directory + pdb_file + '.pdb') # load pdb
    forcefield_solvate = app.ForceField(main_forcefield, water_model_solvate) # specify forcefield and water model
    modeller = app.Modeller(pdb.topology, pdb.positions) # initiate system
    # add water
    modeller.addSolvent(forcefield_solvate, padding=1.5*unit.nanometers, model='tip3p', ionicStrength=0.15*unit.molar)
    # write system
    app.PDBFile.writeFile(modeller.topology, modeller.positions, open(directory + pdb_file + '_solvated.pdb', 'w'))
    # make directories
    os.mkdir(directory + 'mds')
    os.mkdir(directory + 'mds_prep')
# setup solvated system
pdb = app.PDBFile(directory + pdb_file + '_solvated.pdb') # load pdb of solvated system
forcefield = app.ForceField(main_forcefield, water_model) # specify forcefield and water model
modeller = app.Modeller(pdb.topology, pdb.positions) # initiate system
modeller.addExtraParticles(forcefield)
app.PDBFile.writeFile(modeller.topology, modeller.positions, open(directory + '/mds/0.pdb', 'w'))
# run mds
for counter in range(passed_mds, number_mds):
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=1*unit.nanometer,
                                     constraints=app.HBonds) # create system
    # define custom force to restrain atoms
    force = mm.CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    force.addGlobalParameter("k", 5.0 * unit.kilocalories_per_mole / unit.angstroms ** 2)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    # protein heavy atom restrain
    if restrain_heavy_atoms:
        for index, atom in enumerate(modeller.topology.atoms()):
            if atom.residue.name not in non_protein_resnames:
                if atom.element is not None:
                    if atom.element.name is not 'hydrogen':
                        force.addParticle(index, modeller.positions[index].value_in_unit(unit.nanometers))
        system.addForce(force)
    # custom atom restrains
    if len(restrain_ids) > 0:
        for index in restrain_ids:
            force.addParticle(index, modeller.positions[index].value_in_unit(unit.nanometers))
        system.addForce(force)
    integrator = mm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, 2.0*unit.femtoseconds) # 2 fs time step
    simulation = app.Simulation(modeller.topology, system, integrator) # initiate simulation
    simulation.context.setPositions(modeller.positions) # set positions
    simulation.minimizeEnergy() # minimize
    # frame written to dcd each 2500 steps
    simulation.reporters.append(app.DCDReporter(directory + 'mds/' + str(counter) + '.dcd', 2500,
                                                enforcePeriodicBox=False))
    simulation.reporters.append(app.StateDataReporter(stdout, 2500, step=True, potentialEnergy=True, temperature=True,
                                                      speed=True))
    simulation.step((md_length * 1000000) / 2.0)
    os.system("vmd -f mds/0.pdb mds/{0}.dcd -dispdev text -e md_prep.tcl -eofexit -args {0}".format(counter))
