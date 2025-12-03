from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM, Transmon
from quam_libs.macros import qua_declaration, active_reset, readout_state
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from quam_libs.lib.fit import fit_decay_exp, decay_exp
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ['q6']
    use_state_discrimination: bool = True
    use_strict_timing: bool = False
    num_shots: int = 1 
    seed: int = 345324
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    timeout: int = 100
    load_data_id: Optional[int] = None    

params = Parameters()
node = QualibrationNode(name='Single_qubit_training_data_collection', parameters=params)

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations

config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

# %% {QUA_program_parameters}
reset_type = node.parameters.reset_type_thermal_or_active
seed = node.parameters.seed  # Pseudo-random number generator seed
# Flag to enable state discrimination if the readout has been calibrated (rotated blobs and threshold)
state_discrimination = node.parameters.use_state_discrimination
strict_timing = node.parameters.use_strict_timing

preparations = ['0', '1', '+', '-', '+i', '-i'] # Cardinal states
measurement_bases = ["X", "Y", "Z"] # Measurement bases
time = np.linspace(0, 8, 200) # Evolution times in microseconds
n = params.num_shots # Number of shots per configuration

results = []

def prepare_cardinal_state(prep: str, qubit: Transmon):
    if reset_type == "active":
        active_reset(qubit, "readout")
    else:
        qubit.resonator.wait(qubit.thermalization_time * u.ns)

    with switch_(prep):
        with case_('0'):
            pass
        with case_('1'):
            X(qubit)
        with case_('+'):
            H(qubit)
        with case_('-'):
            X(qubit)
            H(qubit)
        with case_('+i'):
            RX(qubit, np.pi/2)
        with case_('-i'):
            RX(qubit, -np.pi/2)

def shift_to_basis(basis: str, qubit: Transmon):
    with switch_(basis):
        with case_('X'):
            H(qubit)
        with case_('Y'):
            RX(qubit, -np.pi/2)
        with case_('Z'):
            pass

with program() as training_data_collection:
    for prep in preparations:
        for basis in measurement_bases:
            for shot in range(n):
                for t in time:
                    for i, qubit in enumerate(qubits):
                        # 1. Prepare cardinal state
                        prepare_cardinal_state(prep, qubit)

                        # 2. Evolve for time t
                        qubit.evolve(t * u.ns)

                        # 3. Shift to measurement basis
                        shift_to_basis(basis)

                        # 4. Acquire weak measurement trace
                        M_t = np.array([0.0])

                        # 5. Perform projective readout (+1 or -1)
                        Y = measure_qubit(basis, qubit)

                        results.append(dict(
                            preparation=prep,
                            base_time=0,
                            evolution_time=t,
                            basis=basis,
                            shot=shot,
                            M_t=M_t,
                            Y=Y
                        ))

node.results = results
node.save()
