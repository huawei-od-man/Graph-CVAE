import os
import pandas as pd
from dataset import QuantumCircuitDataset, TOPO_TYPE_MAP, LAYOUT_METHOD_MAP, ROUTING_METHOD_MAP


def extract_scalars_to_dataframe(dataset: QuantumCircuitDataset) -> pd.DataFrame:
    """
    提取数据集中所有样本的标量指标到DataFrame

    Args:
        dataset: 量子电路数据集实例

    Returns:
        DataFrame包含所有样本的标量指标
    """
    data = []

    # 映射字典（用于将编码值转回原始字符串）
    inv_topo_map = {v: k for k, v in TOPO_TYPE_MAP.items()}
    inv_layout_map = {v: k for k, v in LAYOUT_METHOD_MAP.items()}
    inv_routing_map = {v: k for k, v in ROUTING_METHOD_MAP.items()}

    # 遍历所有样本
    for idx in range(dataset.len()):
        try:
            sample = dataset.get(idx)

            # 提取元信息
            meta = sample["meta"]
            topo_type = inv_topo_map[meta["topo_type"]]
            layout_method = inv_layout_map[meta["layout_method"]]
            routing_method = inv_routing_map[meta["routing_method"]]
            opt_level = meta["optimization_level"]

            # 提取原始电路信息
            origin_info = sample["circuit_origin_info"]

            # 提取优化后电路信息
            opt_info = sample["circuit_optimized_info"]

            # 提取优化指标（不含fidelity）
            metrics = sample["optimization_metrics"]

            # 构造一行数据
            row = {
                # 样本标识
                "sample_idx": idx,
                "base_idx": meta["base_idx"],

                # 拓扑与优化参数
                "topo_type": topo_type,
                "layout_method": layout_method,
                "routing_method": routing_method,
                "optimization_level": opt_level,

                # 原始电路指标
                "origin_num_qubits": origin_info["num_qubits"],
                "origin_total_gates": origin_info["total_gates"],
                "origin_depth": origin_info["depth"],
                "origin_two_qubit_gates": origin_info["two_qubit_gates"],

                # 优化后电路指标
                "opt_num_qubits": opt_info["num_qubits"],
                "opt_total_gates": opt_info["total_gates"],
                "opt_depth": opt_info["depth"],
                "opt_two_qubit_gates": opt_info["two_qubit_gates"],

                # 优化效果指标（不含fidelity）
                "depth_ratio": metrics["depth_ratio"],
                "total_gate_ratio": metrics["total_gate_ratio"],
                "two_qubit_ratio": metrics["two_qubit_ratio"]
            }

            data.append(row)

        except Exception as e:
            print(f"处理样本 {idx} 时出错: {str(e)}")
            continue

    return pd.DataFrame(data)


# 使用示例
if __name__ == "__main__":
    # 加载数据集（使用与生成时相同的参数）
    dataset = QuantumCircuitDataset(
        root="./data",
        base_num_samples=100,  # 与生成时一致
        num_qubits=4,  # 与生成时一致
        max_depth=10,  # 与生成时一致
        regenerate=False  # 不重新生成，仅加载已有数据
    )

    # 提取标量数据到DataFrame
    df = extract_scalars_to_dataframe(dataset)

    # 保存为CSV文件
    output_path = os.path.join(dataset.root, "dataset_scalars.csv")
    df.to_csv(output_path, index=False)
    print(f"标量指标已保存到: {output_path}")

    # 显示部分数据预览
    print("\n数据预览:")
    print(df.head())

    # 简单统计分析示例
    print("\n按优化等级分组的统计:")
    print(df.groupby("optimization_level")[["depth_ratio", "total_gate_ratio"]].mean())

    print("\n按路由方法分组的统计:")
    print(df.groupby("routing_method")[["depth_ratio", "total_gate_ratio"]].mean())
