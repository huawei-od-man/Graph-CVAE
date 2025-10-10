import random
import subprocess
import time
from typing import Union, List
from joblib import Parallel, delayed
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from qiskit.transpiler import CouplingMap
from qiskit import transpile
from qiskit import QuantumCircuit
from dataclasses import dataclass
import math
import itertools
import sys
from topologies import generate_coupling_map


@dataclass
class CircuitStats:
    """
    Obtain statistics of circuits more conveniently
    """
    name: str
    num_qubits: int
    depth: int
    num_gates: int
    num_cx: int
    gate_set: List[str]
    qc: QuantumCircuit

    def __init__(self, qc: QuantumCircuit, name: str):
        self.name = name
        self.depth = qc.depth()
        self.num_gates = sum(qc.count_ops().values())
        self.num_qubits = qc.num_qubits
        self.num_cx = qc.count_ops().get('cx', 0)
        self.gate_set = list(qc.count_ops().keys())
        self.qc = qc

    @classmethod
    def from_qasm_file(cls, qc_path: Path):
        qc_path = Path(qc_path)
        assert qc_path.is_file(), qc_path
        qc = QuantumCircuit.from_qasm_file(str(qc_path))
        return cls(qc, qc_path.stem)


@dataclass
class TranspileStats:
    circuit_name: str
    graph_model: str
    depth_ratio: float
    gate_ratio: float
    cx_ratio: float

    def __init__(self, qc_before: CircuitStats, qc_after: CircuitStats, graph_model: str, transpile_param: dict):
        assert qc_before.name == qc_after.name
        self.circuit_name = qc_before.name
        self.graph_model = graph_model
        self.depth_ratio = round(qc_after.depth / qc_before.depth, 4)
        self.gate_ratio = round(qc_after.num_gates / qc_before.num_gates, 4)
        self.cx_ratio = round(qc_after.num_cx / qc_before.num_cx, 4)
        self.__dict__.update(transpile_param)



def transpile_circuit_on_random_topology(circuit_path: Path,
                                         gate_set,
                                         graph_model: str,
                                         opt_method: str,
                                         layout_method: str = 'sabre',
                                         routing_method: str = 'sabre',
                                         ecc_file=None,
                                         max_iterations=10,
                                         verbose=True,
                                         ):
    """Transpile to a physical coupling graph without optimization (right now)"""
    circuit_path = Path(circuit_path)

    assert circuit_path.is_file(), circuit_path
    assert layout_method in SUPPORTED_LAYOUT_METHOD, layout_method
    assert routing_method in SUPPORTED_ROUTING_METHOD, routing_method

    optimization_level = 0

    if opt_method.startswith('qiskit'):  # qiskit:1 means using qiskit's builtin opt level 1
        optimization_level = int(opt_method.split(':')[-1])
        assert optimization_level in SUPPORTED_OPT_LEVEL
        qc_before = CircuitStats.from_qasm_file(circuit_path)
    elif opt_method == 'quartz':
        assert ecc_file and gate_set
        qc_opt = quartz_optimize(circuit_path, gate_set, ecc_file, verbose)
        qc_before = CircuitStats(qc_opt, name=circuit_path.stem)
    elif opt_method == 'quarl':
        assert ecc_file and gate_set and max_iterations
        qc_opt = quarl_optimize(circuit_path, gate_set, ecc_file, max_iterations=max_iterations,
                                 verbose=verbose)
        qc_before = CircuitStats(qc_opt, name=circuit_path.stem)
    else:
        raise ValueError(opt_method)

    coupling_map = generate_coupling_map(graph_model, num_qubits=qc_before.num_qubits)

    if verbose:
        print(f'circuit {qc_before.name} optimization_level={optimization_level} layout_method={layout_method} routing_method={routing_method}')

    qc_after = transpile(qc_before.qc,
                         basis_gates=gate_set,
                         coupling_map=coupling_map,
                         optimization_level=optimization_level,
                         layout_method=layout_method,
                         routing_method=routing_method,
                         )
    qc_after = CircuitStats(qc_after, qc_before.name)
    transpile_stats = TranspileStats(qc_before, qc_after, graph_model,
                                     transpile_param=dict(opt_method=opt_method,
                                                          layout_method=layout_method,
                                                          routing_method=routing_method))
    return transpile_stats


def dict_product(input_dict):
    keys = input_dict.keys()
    value_lists = input_dict.values()

    # 使用itertools.product生成所有值的组合
    value_combinations = itertools.product(*value_lists)

    # 将每个值的组合与键配对，生成字典列表
    result = [dict(zip(keys, combo)) for combo in value_combinations]

    return result


def make_rounds(circuit_dir: Path, opt_methods: list, repeats=3, verbose=False):
    assert circuit_dir.is_dir(), circuit_dir
    circuit_paths = list(circuit_dir.glob('*.qasm'))

    assert len(circuit_paths), f'No qasm file found in {circuit_dir}'
    rounds = dict_product({
        'circuit_path': circuit_paths,
        'graph_model': SUPPORTED_GRAPH_MODEL,
        'opt_method': opt_methods,
    }) * repeats

    if verbose:
        print('Circuit dir', circuit_dir)
        print('No. circuits', len(circuit_paths))
        print('Opt methods:', opt_methods)
        print('Repeats', repeats)
        print('Make rounds:', len(rounds))

    return rounds


def quarl_optimize(qc_path: Path, gate_set, ecc_file,
                   max_iterations = 50,
                   verbose=True) -> QuantumCircuit:

    gate_set_arg = '[' + ",".join(gate_set) + ']'
    best_graph_output_dir = Path(f'./best_graph/{time.time()}').absolute()
    best_graph_output_dir.mkdir(exist_ok=True, parents=True)
    result_qasm_file = best_graph_output_dir / qc_path.name

    if result_qasm_file.exists():
        result_qasm_file.unlink()

    cmd = f'{sys.executable} ppo.py c.input_graphs.0.name={qc_path.stem} c.input_graphs.0.path={qc_path.absolute()} c.ecc_file={ecc_file} c.gate_set={gate_set_arg} c.max_iterations={max_iterations} c.best_graph_output_dir={best_graph_output_dir}'.split()
    if verbose:
        print(cmd)

    working_dir = Path('experiment/ppo-new/')
    subprocess.check_call(cmd, cwd=working_dir)

    qc = QuantumCircuit.from_qasm_file(result_qasm_file)
    return qc


def quartz_optimize(qasm_file: Path, gate_set, ecc_file, verbose=True) -> QuantumCircuit:
    qasm_file = Path(qasm_file)
    assert qasm_file.is_file(), qasm_file
    ecc_file = Path(ecc_file)
    assert ecc_file.is_file(), ecc_file

    from quartz import Graph, Context

    context = Context(gate_set, ecc_file, verbose=verbose)
    g = Graph.from_qasm_file(context, qasm_file, verbose=verbose)
    g = g.greedy_optimize(ecc_file, verbose=verbose)
    qasm_str = g.to_qasm_str()

    return QuantumCircuit.from_qasm_str(qasm_str)


def run_transpile_and_save_results(circuit_dir: Path, opt_methods: list,
                                    save_file: Path,
                                    gate_set, ecc_file,
                                    n_jobs=-1,
                                    verbose=True):
    rounds = make_rounds(circuit_dir, opt_methods, verbose=verbose)

    results = Parallel(n_jobs=n_jobs, verbose=1)(delayed(transpile_circuit_on_random_topology)(
        gate_set=gate_set,
        ecc_file=ecc_file,
        **rnd,
    ) for rnd in rounds)

    df = pd.DataFrame.from_records(rec.__dict__ for rec in results)
    save_file = Path(save_file)
    save_dir = save_file.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_file, index=False)
    if verbose:
        print(f"Done")
