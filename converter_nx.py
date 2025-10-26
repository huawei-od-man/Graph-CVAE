from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import dag_to_circuit, circuit_to_dag
import networkx as nx


# 1. Qiskit → NetworkX（节点只存 idx，边带 wire）
def qiskit_to_nx_fixed(circuit):
    dag = circuit_to_dag(circuit)
    G = nx.DiGraph()
    for node in dag.op_nodes():
        # 只存索引，不存寄存器名
        G.add_node(node._node_id,
                   name=node.op.name,
                   params=list(node.op.params) if node.op.params else [],
                   qargs=[q.index for q in node.qargs],   # ← int list
                   cargs=[c.index for c in node.cargs])   # ← int list
    # 边：wire 也只存索引（Qubit→int, Clbit→int+offset）
    for src, dst, edge_data in dag.edges():
        wire = edge_data['wire']
        # 统一用 int 表示 wire：量子位=index，经典位=index+max_q
        if hasattr(wire, 'index'):          # Qubit
            wire_id = wire.index
        else:                               # Clbit
            wire_id = wire.index + dag.num_qubits()
        G.add_edge(src._node_id, dst._node_id, wire=wire_id)
    return G


# 2. NetworkX → Qiskit（固定 q/c 寄存器）
def nx_to_qiskit_fixed(G: nx.DiGraph) -> DAGCircuit:
    dag = DAGCircuit()
    # 一次性建寄存器（名字固定）
    max_q = max((max(attr['qargs']) for _, attr in G.nodes(data=True) if attr['qargs']), default=-1) + 1
    max_c = max((max(attr['cargs']) for _, attr in G.nodes(data=True) if attr['cargs']), default=-1) + 1
    qr = QuantumRegister(max_q, 'q')
    cr = ClassicalRegister(max_c, 'c')
    dag.add_qreg(qr)
    dag.add_creg(cr)

    # 按拓扑序加门
    for node_id in nx.topological_sort(G):
        attr = G.nodes[node_id]
        name   = attr['name']
        params = attr['params']
        qidx   = attr['qargs']          # list[int]
        cidx   = attr['cargs']          # list[int]

        # 重建门（示例）
        if name == 'rx' and len(params) == 1:
            gate = RXGate(params[0])
        elif name == 'cx':
            gate = CXGate()
        else:
            raise ValueError(f"unsupported gate: {name}")

        dag.apply_operation_back(gate,
                                  [qr[i] for i in qidx],
                                  [cr[i] for i in cidx])
    return dag

