import sys
import types
import importlib.util
from pathlib import Path
import numpy as np

# Stub out the qtpy dependency required by components.utilities
qtpy_stub = types.ModuleType("qtpy")
qtpy_stub.QtWidgets = types.SimpleNamespace(
    QMessageBox=types.SimpleNamespace(critical=lambda *args, **kwargs: None)
)
sys.modules.setdefault("qtpy", qtpy_stub)

# Dynamically load the utilities module to avoid importing the entire package
utilities_path = Path(__file__).resolve().parents[1] / "components" / "utilities.py"
spec = importlib.util.spec_from_file_location("utilities", utilities_path)
utilities = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utilities)

coherence = utilities.coherence
db2scale = utilities.db2scale
scale2db = utilities.scale2db
rms_time = utilities.rms_time


def test_db_scale_conversions():
    dB = 20.0
    scale = db2scale(dB)
    assert np.isclose(scale, 10.0)
    assert np.isclose(scale2db(scale), dB)


def test_coherence_full_matrix_and_single_pair():
    cpsd = np.zeros((1, 2, 2), dtype=complex)
    cpsd[0, 0, 0] = 4
    cpsd[0, 1, 1] = 4
    cpsd[0, 0, 1] = cpsd[0, 1, 0] = 4
    coh_full = coherence(cpsd)
    assert np.allclose(coh_full, np.ones((1, 2, 2)))
    coh_pair = coherence(cpsd, (0, 1))
    assert np.allclose(coh_pair, np.ones(1))


def test_rms_time():
    signal = np.array([1, -1, 1, -1])
    assert np.isclose(rms_time(signal), 1.0)
    assert np.allclose(rms_time(signal, axis=0), 1.0)
