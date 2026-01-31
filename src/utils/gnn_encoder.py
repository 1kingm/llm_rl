"""GNN encoder for cross-domain topology embedding.

实现真正的图神经网络编码器，支持 GraphSAGE、GAT、GCN 三种模型。
当 PyTorch Geometric 不可用时，回退到统计特征编码。

接入方式:
=========
1. 在 HiPPOCoordinator 中初始化:
   ```python
   from ..utils.gnn_encoder import GNNEncoder, GNNConfig, TopologyGraph

   class HiPPOCoordinator:
       def __init__(self, config, global_agent, local_agent, gnn_config=None):
           self.gnn_encoder = GNNEncoder(gnn_config or GNNConfig())
   ```

2. 在 select_action 中使用:
   ```python
   def select_action(self, state_high, state_low, network_state=None, domain_loads=None):
       if network_state is not None:
           graph = TopologyGraph.from_network_state(network_state, domain_loads)
           h_topo = self.gnn_encoder.encode(graph)
           state_high = list(state_high) + h_topo
       ...
   ```

3. 在 rollout.py 中传递 network_state:
   ```python
   action = coordinator.select_action(
       state_high, state_low,
       network_state=env.current_network_state
   )
   ```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .types import NetworkState

logger = logging.getLogger(__name__)

# 尝试导入 PyTorch 和 PyTorch Geometric
_TORCH_AVAILABLE = False
_PYG_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None

try:
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool

    _PYG_AVAILABLE = True
except ImportError:
    torch_geometric = None
    Data = None
    GATConv = None
    GCNConv = None
    SAGEConv = None
    global_mean_pool = None


@dataclass(frozen=True)
class GNNConfig:
    """GNN 编码器配置."""

    model_type: str = "GraphSAGE"  # GraphSAGE, GAT, GCN
    num_layers: int = 2
    hidden_dim: int = 64
    out_dim: int = 32
    aggregation: str = "mean"  # mean, max, sum (for GraphSAGE)
    dropout: float = 0.1
    use_edge_features: bool = True
    # GAT 特定参数
    num_heads: int = 4
    # 训练参数
    learning_rate: float = 1e-3
    # 是否强制使用统计编码（用于调试）
    force_statistical: bool = False


@dataclass
class TopologyGraph:
    """跨域网络拓扑图表示.

    节点: 域 v_k
    边: 域间链路 e_{ij}
    节点特征: [U_k^{GPU}, U_k^{Mem}, Q_k]
    边特征: [B_{ij}(t), L_{ij}(t)]
    """

    node_features: List[List[float]]
    edge_index: List[List[int]]  # [src_nodes, dst_nodes]
    edge_features: Optional[List[List[float]]] = None
    num_nodes: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "num_nodes", len(self.node_features))

    @classmethod
    def from_network_state(
        cls,
        network_state: NetworkState,
        domain_loads: Optional[List[float]] = None,
        mem_utils: Optional[List[float]] = None,
        queue_lengths: Optional[List[float]] = None,
    ) -> "TopologyGraph":
        """从 NetworkState 构建拓扑图.

        Args:
            network_state: 网络状态（带宽/延迟矩阵）
            domain_loads: 各域 GPU 利用率 [U_k^{GPU}]
            mem_utils: 各域内存利用率 [U_k^{Mem}]
            queue_lengths: 各域任务队列长度 [Q_k]

        Returns:
            TopologyGraph 实例
        """
        num_domains = len(network_state.bandwidth_gbps)

        # 计算带宽和延迟的统计量用于归一化
        bw_values = []
        lat_values = []
        for i in range(num_domains):
            for j in range(num_domains):
                if i != j:
                    bw_values.append(network_state.bandwidth_gbps[i][j])
                    lat_values.append(network_state.latency_ms[i][j])

        bw_max = max(bw_values) if bw_values else 1.0
        lat_max = max(lat_values) if lat_values else 1.0

        # 构建全连接图的边索引（不含自环）
        src_nodes: List[int] = []
        dst_nodes: List[int] = []
        edge_feats: List[List[float]] = []

        for i in range(num_domains):
            for j in range(num_domains):
                if i != j:
                    src_nodes.append(i)
                    dst_nodes.append(j)
                    # 边特征: [带宽(归一化), 延迟(归一化)]
                    bw = network_state.bandwidth_gbps[i][j] / bw_max
                    lat = network_state.latency_ms[i][j] / lat_max
                    edge_feats.append([bw, lat])

        # 节点特征: 使用提供的值或默认值
        if domain_loads is None:
            domain_loads = [0.5] * num_domains
        if mem_utils is None:
            mem_utils = [0.5] * num_domains
        if queue_lengths is None:
            queue_lengths = [0.5] * num_domains

        # 节点特征: [GPU利用率, 内存利用率, 队列长度(归一化)]
        node_feats = [
            [domain_loads[i], mem_utils[i], queue_lengths[i]]
            for i in range(num_domains)
        ]

        return cls(
            node_features=node_feats,
            edge_index=[src_nodes, dst_nodes],
            edge_features=edge_feats,
        )


if _TORCH_AVAILABLE and _PYG_AVAILABLE:

    class GraphSAGEModel(nn.Module):
        """GraphSAGE 模型实现."""

        def __init__(self, config: GNNConfig, in_channels: int, edge_dim: int = 2):
            super().__init__()
            self.config = config
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()

            # 第一层
            self.convs.append(
                SAGEConv(in_channels, config.hidden_dim, aggr=config.aggregation)
            )
            self.norms.append(nn.LayerNorm(config.hidden_dim))

            # 中间层
            for _ in range(config.num_layers - 2):
                self.convs.append(
                    SAGEConv(config.hidden_dim, config.hidden_dim, aggr=config.aggregation)
                )
                self.norms.append(nn.LayerNorm(config.hidden_dim))

            # 最后一层
            if config.num_layers > 1:
                self.convs.append(
                    SAGEConv(config.hidden_dim, config.out_dim, aggr=config.aggregation)
                )

            # 边特征投影（如果使用边特征）
            if config.use_edge_features:
                self.edge_proj = nn.Linear(edge_dim, config.hidden_dim)

            self.dropout = config.dropout

        def forward(self, x, edge_index, edge_attr=None, batch=None):
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index)
                x = self.norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.convs[-1](x, edge_index)

            # 全局池化
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            x = global_mean_pool(x, batch)

            return x

    class GATModel(nn.Module):
        """Graph Attention Network 模型实现."""

        def __init__(self, config: GNNConfig, in_channels: int, edge_dim: int = 2):
            super().__init__()
            self.config = config
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()

            # 第一层
            self.convs.append(
                GATConv(
                    in_channels,
                    config.hidden_dim // config.num_heads,
                    heads=config.num_heads,
                    dropout=config.dropout,
                    edge_dim=edge_dim if config.use_edge_features else None,
                )
            )
            self.norms.append(nn.LayerNorm(config.hidden_dim))

            # 中间层
            for _ in range(config.num_layers - 2):
                self.convs.append(
                    GATConv(
                        config.hidden_dim,
                        config.hidden_dim // config.num_heads,
                        heads=config.num_heads,
                        dropout=config.dropout,
                        edge_dim=edge_dim if config.use_edge_features else None,
                    )
                )
                self.norms.append(nn.LayerNorm(config.hidden_dim))

            # 最后一层（单头）
            if config.num_layers > 1:
                self.convs.append(
                    GATConv(
                        config.hidden_dim,
                        config.out_dim,
                        heads=1,
                        concat=False,
                        dropout=config.dropout,
                        edge_dim=edge_dim if config.use_edge_features else None,
                    )
                )

            self.dropout = config.dropout

        def forward(self, x, edge_index, edge_attr=None, batch=None):
            for i, conv in enumerate(self.convs[:-1]):
                if self.config.use_edge_features and edge_attr is not None:
                    x = conv(x, edge_index, edge_attr=edge_attr)
                else:
                    x = conv(x, edge_index)
                x = self.norms[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            if self.config.use_edge_features and edge_attr is not None:
                x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
            else:
                x = self.convs[-1](x, edge_index)

            # 全局池化
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            x = global_mean_pool(x, batch)

            return x

    class GCNModel(nn.Module):
        """Graph Convolutional Network 模型实现."""

        def __init__(self, config: GNNConfig, in_channels: int, edge_dim: int = 2):
            super().__init__()
            self.config = config
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()

            # 第一层
            self.convs.append(GCNConv(in_channels, config.hidden_dim))
            self.norms.append(nn.LayerNorm(config.hidden_dim))

            # 中间层
            for _ in range(config.num_layers - 2):
                self.convs.append(GCNConv(config.hidden_dim, config.hidden_dim))
                self.norms.append(nn.LayerNorm(config.hidden_dim))

            # 最后一层
            if config.num_layers > 1:
                self.convs.append(GCNConv(config.hidden_dim, config.out_dim))

            self.dropout = config.dropout

        def forward(self, x, edge_index, edge_attr=None, batch=None):
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index)
                x = self.norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.convs[-1](x, edge_index)

            # 全局池化
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            x = global_mean_pool(x, batch)

            return x


class GNNEncoder:
    """GNN 拓扑编码器.

    将跨域网络拓扑编码为固定维度的向量 h_topo，
    作为 GlobalAgent 的额外输入特征。

    支持三种模式：
    1. GraphSAGE: 采样聚合，适合大规模图
    2. GAT: 注意力机制，适合异构图
    3. GCN: 谱卷积，适合同构图

    当 PyTorch Geometric 不可用时，自动回退到统计特征编码。
    """

    def __init__(self, config: GNNConfig | None = None) -> None:
        self.config = config or GNNConfig()
        self._model: Optional[nn.Module] = None
        self._device = "cpu"
        self._use_gnn = False

        # 初始化 GNN 模型
        if (
            _TORCH_AVAILABLE
            and _PYG_AVAILABLE
            and not self.config.force_statistical
        ):
            self._init_gnn_model()
        else:
            if not _TORCH_AVAILABLE:
                logger.warning("PyTorch 不可用，回退到统计特征编码")
            elif not _PYG_AVAILABLE:
                logger.warning("PyTorch Geometric 不可用，回退到统计特征编码")
            elif self.config.force_statistical:
                logger.info("强制使用统计特征编码")

    def _init_gnn_model(self) -> None:
        """初始化 GNN 模型."""
        in_channels = 3  # 节点特征维度: [GPU利用率, 内存利用率, 队列长度]
        edge_dim = 2  # 边特征维度: [带宽, 延迟]

        model_type = self.config.model_type.lower()
        if model_type in {"graphsage", "gcn"} and self.config.use_edge_features:
            logger.warning(
                "%s 不支持 edge features，将忽略 edge features 配置。", self.config.model_type
            )
        if model_type == "graphsage":
            self._model = GraphSAGEModel(self.config, in_channels, edge_dim)
        elif model_type == "gat":
            self._model = GATModel(self.config, in_channels, edge_dim)
        elif model_type == "gcn":
            self._model = GCNModel(self.config, in_channels, edge_dim)
        else:
            logger.warning(f"未知模型类型 {self.config.model_type}，使用 GraphSAGE")
            self._model = GraphSAGEModel(self.config, in_channels, edge_dim)

        self._model.eval()
        self._use_gnn = True
        logger.info(f"GNN 编码器初始化完成: {self.config.model_type}")

    def has_model(self) -> bool:
        """是否具备可训练的 GNN 模型."""
        return self._use_gnn and self._model is not None

    def to(self, device: str) -> "GNNEncoder":
        """将模型移动到指定设备."""
        self._device = device
        if self._model is not None:
            self._model = self._model.to(device)
        return self

    def train(self) -> None:
        """切换到训练模式."""
        if self._model is not None:
            self._model.train()

    def eval(self) -> None:
        """切换到评估模式."""
        if self._model is not None:
            self._model.eval()

    def parameters(self):
        """获取模型参数（用于优化器注册）。"""
        if self._model is None:
            return []
        return self._model.parameters()

    def state_dict(self) -> Optional[Dict[str, Any]]:
        """获取模型权重."""
        if self._model is None:
            return None
        return self._model.state_dict()

    def load_state_dict(self, state_dict: Optional[Dict[str, Any]]) -> None:
        """加载模型权重."""
        if self._model is None or state_dict is None:
            return
        self._model.load_state_dict(state_dict)

    def encode(self, graph: TopologyGraph) -> List[float]:
        """将拓扑图编码为固定长度向量.

        Args:
            graph: TopologyGraph 实例

        Returns:
            h_topo: 拓扑编码向量，维度为 config.out_dim
        """
        if self._use_gnn and self._model is not None:
            return self._gnn_encode(graph)
        return self._statistical_encode(graph)

    def encode_tensor(self, graph: TopologyGraph, with_grad: bool = False) -> torch.Tensor:
        """返回 torch.Tensor 形式的编码（可选保留梯度）."""
        if torch is None:
            raise RuntimeError("PyTorch 不可用，无法返回 tensor 编码。")
        if self._use_gnn and self._model is not None:
            return self._gnn_encode_tensor(graph, with_grad=with_grad)
        stats = self._statistical_encode(graph)
        return torch.tensor(stats, dtype=torch.float32, device=self._device)

    def _gnn_encode(self, graph: TopologyGraph) -> List[float]:
        """使用 GNN 模型编码."""
        h = self._gnn_encode_tensor(graph, with_grad=False)
        return h.cpu().tolist()

    def _gnn_encode_tensor(self, graph: TopologyGraph, with_grad: bool = False) -> torch.Tensor:
        """使用 GNN 模型编码（tensor 输出）."""
        # 转换为 PyTorch Geometric Data 对象
        x = torch.tensor(graph.node_features, dtype=torch.float32, device=self._device)
        edge_index = torch.tensor(graph.edge_index, dtype=torch.long, device=self._device)

        edge_attr = None
        if graph.edge_features and self.config.use_edge_features:
            edge_attr = torch.tensor(
                graph.edge_features, dtype=torch.float32, device=self._device
            )

        # 前向传播
        if with_grad:
            h = self._model(x, edge_index, edge_attr=edge_attr)
        else:
            with torch.no_grad():
                h = self._model(x, edge_index, edge_attr=edge_attr)

        return h.squeeze(0)

    def encode_from_network_state(
        self,
        network_state: NetworkState,
        domain_loads: Optional[List[float]] = None,
        mem_utils: Optional[List[float]] = None,
        queue_lengths: Optional[List[float]] = None,
    ) -> List[float]:
        """便捷方法：直接从 NetworkState 编码.

        Args:
            network_state: 网络状态
            domain_loads: 各域 GPU 利用率
            mem_utils: 各域内存利用率
            queue_lengths: 各域任务队列长度

        Returns:
            h_topo: 拓扑编码向量
        """
        graph = TopologyGraph.from_network_state(
            network_state, domain_loads, mem_utils, queue_lengths
        )
        return self.encode(graph)

    def encode_tensor_from_network_state(
        self,
        network_state: NetworkState,
        domain_loads: Optional[List[float]] = None,
        mem_utils: Optional[List[float]] = None,
        queue_lengths: Optional[List[float]] = None,
        with_grad: bool = False,
    ) -> torch.Tensor:
        """便捷方法：直接从 NetworkState 返回 tensor 编码."""
        graph = TopologyGraph.from_network_state(
            network_state, domain_loads, mem_utils, queue_lengths
        )
        return self.encode_tensor(graph, with_grad=with_grad)

    def _statistical_encode(self, graph: TopologyGraph) -> List[float]:
        """统计特征编码（回退方案）.

        当 PyTorch Geometric 不可用时使用。
        提取图的统计特征作为编码。
        """
        out_dim = self.config.out_dim
        if not graph.node_features:
            return [0.0] * out_dim

        node_feats = graph.node_features
        edge_feats = graph.edge_features or []

        # 节点特征统计
        node_means = [sum(vals) / len(vals) for vals in zip(*node_feats)]
        node_stds = self._std_per_feature(node_feats, node_means)
        node_mins = [min(vals) for vals in zip(*node_feats)]
        node_maxs = [max(vals) for vals in zip(*node_feats)]

        # 边特征统计
        if edge_feats:
            edge_means = [sum(vals) / len(vals) for vals in zip(*edge_feats)]
            edge_stds = self._std_per_feature(edge_feats, edge_means)
            edge_mins = [min(vals) for vals in zip(*edge_feats)]
            edge_maxs = [max(vals) for vals in zip(*edge_feats)]
        else:
            edge_means = [0.0, 0.0]
            edge_stds = [0.0, 0.0]
            edge_mins = [0.0, 0.0]
            edge_maxs = [0.0, 0.0]

        # 负载不均衡度
        load_values = [feat[0] for feat in node_feats]
        load_cv = self._coefficient_of_variation(load_values)
        load_gini = self._gini_coefficient(load_values)

        # 图结构特征
        num_nodes = float(graph.num_nodes)
        num_edges = float(len(graph.edge_index[0])) if graph.edge_index else 0.0
        density = num_edges / max(1.0, num_nodes * max(1.0, num_nodes - 1.0))

        # 带宽异构性
        bw_values = [e[0] for e in edge_feats] if edge_feats else []
        bw_cv = self._coefficient_of_variation(bw_values)

        # 延迟异构性
        lat_values = [e[1] for e in edge_feats] if edge_feats else []
        lat_cv = self._coefficient_of_variation(lat_values)

        # 组合所有特征
        features = (
            node_means
            + node_stds
            + node_mins
            + node_maxs
            + edge_means
            + edge_stds
            + edge_mins
            + edge_maxs
            + [load_cv, load_gini, density, bw_cv, lat_cv, num_nodes / 10.0]
        )

        # 填充或截断到目标维度
        if len(features) < out_dim:
            features.extend([0.0] * (out_dim - len(features)))
        else:
            features = features[:out_dim]

        return features

    @staticmethod
    def _std_per_feature(
        values: List[List[float]], means: List[float]
    ) -> List[float]:
        """计算每个特征的标准差（使用无偏估计）."""
        stds: List[float] = []
        n = len(values)
        for idx, mean in enumerate(means):
            if n <= 1:
                stds.append(0.0)
            else:
                # 使用 Bessel 校正的无偏估计
                variance = sum((row[idx] - mean) ** 2 for row in values) / (n - 1)
                stds.append(variance ** 0.5)
        return stds

    @staticmethod
    def _coefficient_of_variation(values: List[float]) -> float:
        """计算变异系数 (CV = std / mean)."""
        if not values or len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        if mean == 0:
            return 0.0
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        std = variance ** 0.5
        return std / mean

    @staticmethod
    def _gini_coefficient(values: List[float]) -> float:
        """计算基尼系数，衡量不均衡度."""
        if not values:
            return 0.0
        n = len(values)
        if n == 1:
            return 0.0
        sorted_values = sorted(values)
        total = sum(sorted_values)
        if total == 0:
            return 0.0
        cumsum = 0.0
        gini_sum = 0.0
        for i, v in enumerate(sorted_values):
            cumsum += v
            gini_sum += cumsum - v / 2
        return 1 - 2 * gini_sum / (n * total)

    def train_step(
        self,
        graphs: List[TopologyGraph],
        targets: List[List[float]],
        optimizer: Optional[Any] = None,
    ) -> float:
        """训练一步（用于端到端训练）.

        Args:
            graphs: 拓扑图列表
            targets: 目标编码列表
            optimizer: PyTorch 优化器

        Returns:
            loss: 训练损失
        """
        if not self._use_gnn or self._model is None:
            return 0.0

        if optimizer is None:
            optimizer = torch.optim.Adam(
                self._model.parameters(), lr=self.config.learning_rate
            )

        self._model.train()
        total_loss = 0.0

        for graph, target in zip(graphs, targets):
            x = torch.tensor(
                graph.node_features, dtype=torch.float32, device=self._device
            )
            edge_index = torch.tensor(
                graph.edge_index, dtype=torch.long, device=self._device
            )
            edge_attr = None
            if graph.edge_features and self.config.use_edge_features:
                edge_attr = torch.tensor(
                    graph.edge_features, dtype=torch.float32, device=self._device
                )
            target_tensor = torch.tensor(
                target, dtype=torch.float32, device=self._device
            ).unsqueeze(0)

            optimizer.zero_grad()
            h = self._model(x, edge_index, edge_attr=edge_attr)
            loss = F.mse_loss(h, target_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        self._model.eval()
        return total_loss / max(1, len(graphs))

    def export_config(self) -> Dict[str, Any]:
        """导出配置用于日志记录."""
        return {
            "model_type": self.config.model_type,
            "num_layers": self.config.num_layers,
            "hidden_dim": self.config.hidden_dim,
            "out_dim": self.config.out_dim,
            "aggregation": self.config.aggregation,
            "dropout": self.config.dropout,
            "use_edge_features": self.config.use_edge_features,
            "num_heads": self.config.num_heads,
            "use_gnn": self._use_gnn,
            "device": self._device,
        }

    def save(self, path: str) -> None:
        """保存模型权重."""
        if self._model is not None:
            torch.save(self._model.state_dict(), path)
            logger.info(f"GNN 模型已保存到 {path}")

    def load(self, path: str) -> None:
        """加载模型权重."""
        if self._model is not None:
            self._model.load_state_dict(torch.load(path, map_location=self._device))
            self._model.eval()
            logger.info(f"GNN 模型已从 {path} 加载")
