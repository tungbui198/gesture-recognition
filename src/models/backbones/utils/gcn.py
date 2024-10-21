import copy as cp
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmengine.model import BaseModule, ModuleList, Sequential


class unit_gcn(BaseModule):
    """The basic unit of graph convolutional network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        A (torch.Tensor): The adjacency matrix defined in the graph
            with shape of `(num_subsets, num_nodes, num_nodes)`.
        adaptive (str): The strategy for adapting the weights of the
            adjacency matrix. Defaults to ``'importance'``.
        conv_pos (str): The position of the 1x1 2D conv.
            Defaults to ``'pre'``.
        with_res (bool): Whether to use residual connection.
            Defaults to False.
        norm (str): The name of norm layer. Defaults to ``'BN'``.
        act (str): The name of activation layer. Defaults to ``'Relu'``.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        adaptive: str = "importance",
        conv_pos: str = "pre",
        with_res: bool = False,
        norm: str = "BN",
        act: str = "ReLU",
        init_cfg: Optional[Union[Dict, List[Dict]]] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = A.size(0)

        assert adaptive in [None, "init", "offset", "importance"]
        self.adaptive = adaptive
        assert conv_pos in ["pre", "post"]
        self.conv_pos = conv_pos
        self.with_res = with_res

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]
        self.act = build_activation_layer(self.act_cfg)

        if self.adaptive == "init":
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer("A", A)

        if self.adaptive in ["offset", "importance"]:
            self.PA = nn.Parameter(A.clone())
            if self.adaptive == "offset":
                nn.init.uniform_(self.PA, -1e-6, 1e-6)
            elif self.adaptive == "importance":
                nn.init.constant_(self.PA, 1)

        if self.conv_pos == "pre":
            self.conv = nn.Conv2d(in_channels, out_channels * A.size(0), 1)
        elif self.conv_pos == "post":
            self.conv = nn.Conv2d(A.size(0) * in_channels, out_channels, 1)

        if self.with_res:
            if in_channels != out_channels:
                self.down = Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    build_norm_layer(self.norm_cfg, out_channels)[1],
                )
            else:
                self.down = lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape
        res = self.down(x) if self.with_res else 0

        A_switch = {None: self.A, "init": self.A}
        if hasattr(self, "PA"):
            A_switch.update(
                {"offset": self.A + self.PA, "importance": self.A * self.PA}
            )
        A = A_switch[self.adaptive]

        if self.conv_pos == "pre":
            x = self.conv(x)
            x = x.view(n, self.num_subsets, -1, t, v)
            x = torch.einsum("nkctv,kvw->nctw", (x, A)).contiguous()
        elif self.conv_pos == "post":
            x = torch.einsum("nctv,kvw->nkctw", (x, A)).contiguous()
            x = x.view(n, -1, t, v)
            x = self.conv(x)

        return self.act(self.bn(x) + res)


class aagcn(BaseModule):
    """The graph convolution unit of AAGCN.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        A (torch.Tensor): The adjacency matrix defined in the graph
            with shape of `(num_subsets, num_joints, num_joints)`.
        coff_embedding (int): The coefficient for downscaling the embedding
            dimension. Defaults to 4.
        adaptive (bool): Whether to use adaptive graph convolutional layer.
            Defaults to True.
        attention (bool): Whether to use the STC-attention module.
            Defaults to True.
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
                dict(type='Constant', layer='BatchNorm2d', val=1,
                     override=dict(type='Constant', name='bn', val=1e-6)),
                dict(type='Kaiming', layer='Conv2d', mode='fan_out'),
                dict(type='ConvBranch', name='conv_d')
            ]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        coff_embedding: int = 4,
        adaptive: bool = True,
        attention: bool = True,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(
                type="Constant",
                layer="BatchNorm2d",
                val=1,
                override=dict(type="Constant", name="bn", val=1e-6),
            ),
            dict(type="Kaiming", layer="Conv2d", mode="fan_out"),
            dict(type="ConvBranch", name="conv_d"),
        ],
    ) -> None:

        if attention:
            attention_init_cfg = [
                dict(
                    type="Constant",
                    layer="Conv1d",
                    val=0,
                    override=dict(type="Xavier", name="conv_sa"),
                ),
                dict(
                    type="Kaiming",
                    layer="Linear",
                    mode="fan_in",
                    override=dict(type="Constant", val=0, name="fc2c"),
                ),
            ]
            init_cfg = cp.copy(init_cfg)
            init_cfg.extend(attention_init_cfg)

        super(aagcn, self).__init__(init_cfg=init_cfg)
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        self.attention = attention

        num_joints = A.shape[-1]

        self.conv_d = ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if self.adaptive:
            self.A = nn.Parameter(A)

            self.alpha = nn.Parameter(torch.zeros(1))
            self.conv_a = ModuleList()
            self.conv_b = ModuleList()
            for i in range(self.num_subset):
                self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
                self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
        else:
            self.register_buffer("A", A)

        if self.attention:
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
            # s attention
            ker_joint = num_joints if num_joints % 2 else num_joints - 1
            pad = (ker_joint - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_joint, padding=pad)
            # channel attention
            rr = 2
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)

        self.down = lambda x: x
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
            )

        self.bn = nn.BatchNorm2d(out_channels)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            for i in range(self.num_subset):
                A1 = (
                    self.conv_a[i](x)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                    .view(N, V, self.inter_c * T)
                )
                A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
                A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))  # N V V
                A1 = self.A[i] + A1 * self.alpha
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z
        else:
            for i in range(self.num_subset):
                A1 = self.A[i]
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z

        y = self.relu(self.bn(y) + self.down(x))

        if self.attention:
            # spatial attention first
            se = y.mean(-2)  # N C V
            se1 = self.sigmoid(self.conv_sa(se))  # N 1 V
            y = y * se1.unsqueeze(-2) + y
            # then temporal attention
            se = y.mean(-1)  # N C T
            se1 = self.sigmoid(self.conv_ta(se))  # N 1 T
            y = y * se1.unsqueeze(-1) + y
            # then spatial temporal attention ??
            se = y.mean(-1).mean(-1)  # N C
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))  # N C
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
            # A little bit weird
        return y


class dggcn(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        ratio=0.25,
        ctr="T",
        ada="T",
        subset_wise=False,
        ada_act="softmax",
        ctr_act="tanh",
        norm="BN",
        act="ReLU",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        num_subsets = A.size(0)
        self.num_subsets = num_subsets
        self.ctr = ctr
        self.ada = ada
        self.ada_act = ada_act
        self.ctr_act = ctr_act
        assert ada_act in ["tanh", "relu", "sigmoid", "softmax"]
        assert ctr_act in ["tanh", "relu", "sigmoid", "softmax"]

        self.subset_wise = subset_wise

        assert self.ctr in [None, "NA", "T"]
        assert self.ada in [None, "NA", "T"]

        if ratio is None:
            ratio = 1 / self.num_subsets
        self.ratio = ratio
        mid_channels = int(ratio * out_channels)
        self.mid_channels = mid_channels

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.act = build_activation_layer(self.act_cfg)

        self.A = nn.Parameter(A.clone())

        # Introduce non-linear
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels * num_subsets, 1),
            build_norm_layer(self.norm_cfg, mid_channels * num_subsets)[1],
            self.act,
        )
        self.post = nn.Conv2d(mid_channels * num_subsets, out_channels, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-2)

        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))
        self.beta = nn.Parameter(torch.zeros(self.num_subsets))

        if self.ada or self.ctr:
            self.conv1 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)
            self.conv2 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                build_norm_layer(self.norm_cfg, out_channels)[1],
            )
        else:
            self.down = lambda x: x
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape

        res = self.down(x)
        A = self.A

        # 1 (N), K, 1 (C), 1 (T), V, V
        A = A[None, :, None, None]
        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v)
        # * The shape of pre_x is N, K, C, T, V

        x1, x2 = None, None
        if self.ctr is not None or self.ada is not None:
            # The shape of tmp_x is N, C, T or 1, V
            tmp_x = x

            if not (self.ctr == "NA" or self.ada == "NA"):
                tmp_x = tmp_x.mean(dim=-2, keepdim=True)

            x1 = self.conv1(tmp_x).reshape(
                n, self.num_subsets, self.mid_channels, -1, v
            )
            x2 = self.conv2(tmp_x).reshape(
                n, self.num_subsets, self.mid_channels, -1, v
            )

        if self.ctr is not None:
            # * The shape of ada_graph is N, K, C[1], T or 1, V, V
            diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)
            ada_graph = getattr(self, self.ctr_act)(diff)

            if self.subset_wise:
                ada_graph = torch.einsum("nkctuv,k->nkctuv", ada_graph, self.alpha)
            else:
                ada_graph = ada_graph * self.alpha[0]
            A = ada_graph + A

        if self.ada is not None:
            # * The shape of ada_graph is N, K, 1, T[1], V, V
            ada_graph = torch.einsum("nkctv,nkctw->nktvw", x1, x2)[:, :, None]
            ada_graph = getattr(self, self.ada_act)(ada_graph)

            if self.subset_wise:
                ada_graph = torch.einsum("nkctuv,k->nkctuv", ada_graph, self.beta)
            else:
                ada_graph = ada_graph * self.beta[0]
            A = ada_graph + A

        if self.ctr is not None or self.ada is not None:
            assert len(A.shape) == 6
            # * C, T can be 1
            if A.shape[2] == 1 and A.shape[3] == 1:
                A = A.squeeze(2).squeeze(2)
                x = torch.einsum("nkctv,nkvw->nkctw", pre_x, A).contiguous()
            elif A.shape[2] == 1:
                A = A.squeeze(2)
                x = torch.einsum("nkctv,nktvw->nkctw", pre_x, A).contiguous()
            elif A.shape[3] == 1:
                A = A.squeeze(3)
                x = torch.einsum("nkctv,nkcvw->nkctw", pre_x, A).contiguous()
            else:
                x = torch.einsum("nkctv,nkctvw->nkctw", pre_x, A).contiguous()
        else:
            # * The graph shape is K, V, V
            A = A.squeeze()
            assert len(A.shape) in [2, 3] and A.shape[-2] == A.shape[-1]
            if len(A.shape) == 2:
                A = A[None]
            x = torch.einsum("nkctv,kvw->nkctw", pre_x, A).contiguous()

        x = x.reshape(n, -1, t, v)
        x = self.post(x)
        return self.act(self.bn(x) + res)
