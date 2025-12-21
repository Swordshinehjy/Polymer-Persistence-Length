from . import (chain_rotation, deflection, model, model_dependent,
               model_dihedral, planarity, relaxed_dihedral_scan)
from .model import (PolymerPersistence, compute_persistence_alternating,
                    compute_persistence_terpolymer,
                    compute_persistence_terpolymer_Tscan, inverse_data)
from .model_dependent import PolymerPersistenceDependentDefelection
from .model_dihedral import PolymerPersistenceDependentDihedral
from .planarity import PolymerPlanarity
from .relaxed_dihedral_scan import (GaussianLogParser,
                                    XYZDeflectionAngleCalculator)

__version__ = '0.9.17'
