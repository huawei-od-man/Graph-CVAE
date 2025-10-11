from qiskit.circuit.library import HGate, XGate, CXGate, SwapGate, ZGate, TGate, SGate

GATE_CLS_MAP = {
    'h': HGate, 'x': XGate, 'z': ZGate, 't': TGate, 's': SGate,
    'cx': CXGate, 'swap': SwapGate
}

SUPPORTED_LAYOUT_METHOD = ('trivial', 'dense', 'sabre')
SUPPORTED_ROUTING_METHOD = ('basic', 'sabre')  # lookahead
SUPPORTED_GRAPH_MODEL = ('linear', 'star', 'grid', 'random')
SUPPORTED_OPT_LEVEL = tuple(range(3))  # Exclude 3 (too slow)


# -------------------------- 辅助函数：量子比特索引获取（兼容Qiskit版本） --------------------------
def get_qubit_index(q) -> int:
    """安全获取量子比特的整数索引"""
    if hasattr(q, 'index'):
        return q.index
    elif hasattr(q, '_index'):
        return q._index
    elif hasattr(q, 'register') and hasattr(q.register, 'qubits'):
        return q.register.qubits.index(q)
    else:
        raise AttributeError(f"无法获取比特索引，比特属性：{dir(q)}")
