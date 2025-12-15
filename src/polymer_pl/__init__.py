from . import chain_rotation, deflection, relaxed_dihedral_scan, model, planarity, model_dependent, model_dihedral
from .model import (PolymerPersistence, compute_persistence_alternating,
                    compute_persistence_terpolymer,
                    compute_persistence_terpolymer_Tscan, inverse_data)
from .planarity import PolymerPlanarity
from .model_dependent import PolymerPersistenceDependentDefelection
from .relaxed_dihedral_scan import GaussianLogParser, XYZDeflectionAngleCalculator
from .model_dihedral import PolymerPersistenceDependentDihedral

__version__ = '0.9.15'
