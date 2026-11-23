import json
import numpy as np
import pytest
from app.server import (
    app,
    extract_ca_coords,
    tile_long_sequence,
    verify_topology_3d,
    AA1_TO_AA3,
)
from space.app import ca_to_pdb


def test_extract_ca_coords():
    pdb_sample = (
        "ATOM      1  N   MET A   1      -0.540   1.230   2.340  1.00 75.50           N\n"
        "ATOM      2  CA  MET A   1       0.000   0.000   0.000  1.00 85.20           C\n"
        "ATOM      3  C   MET A   1       1.200   0.450   0.890  1.00 80.10           C\n"
        "ATOM      4  CA  GLN A   2       2.000   0.000   0.000  1.00 92.40           C\n"
    )
    coords, plddts = extract_ca_coords(pdb_sample)
    assert len(coords) == 2
    assert len(plddts) == 2
    assert np.allclose(coords[0], [0.0, 0.0, 0.0])
    assert np.allclose(coords[1], [2.0, 0.0, 0.0])
    assert plddts[0] == 85.20
    assert plddts[1] == 92.40


def test_ca_to_pdb_fidelity():
    coords = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    plddts = [88.50, 92.30]
    seq = "MQ"
    pdb_out = ca_to_pdb(coords, plddts, seq)
    lines = pdb_out.strip().splitlines()
    assert len(lines) == 3  # 2 ATOM + 1 END
    assert "CA  MET A   1" in lines[0]
    assert "88.50" in lines[0]
    assert "CA  GLN A   2" in lines[1]
    assert "92.30" in lines[1]
    assert lines[2] == "END"


def test_verify_topology_3d_cli():
    # Simple linear segment should be Unknot
    ca_coords = [
        {"x": 0.0, "y": 0.0, "z": 0.0},
        {"x": 1.0, "y": 0.0, "z": 0.0},
        {"x": 1.0, "y": 1.0, "z": 0.0},
        {"x": 0.0, "y": 1.0, "z": 0.0},
    ]
    topo = verify_topology_3d(ca_coords)
    assert topo.get("knot_type") == "Unknot (0_1)"
    assert topo.get("is_knotted") is False
    assert topo.get("chain_breaks") == 0


def test_server_health_and_predict_custom():
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["weights_loaded"] is True

    # Test /predict_custom on Ubiquitin
    ubq_seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    resp2 = client.post("/predict_custom", json={"sequence": ubq_seq})
    assert resp2.status_code == 200
    data2 = resp2.get_json()
    assert data2["predicted_knot"] in ["Unknot (0_1)", "Trefoil (3_1)", "Figure-Eight (4_1)", "Complex"]
    assert "probabilities" in data2
    assert "confidence" in data2
    assert data2["weights_loaded"] is True
