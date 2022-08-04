from det3d.core.bbox import box_torch_ops
from ..registry import DETECTORS
from .base import BaseDetector
from .. import builder
import torch 
from torch import nn 
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.parameter import Parameter
from matplotlib import pyplot as plt

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 self_posembed=None, cross_posembed=None, cross_only=False):
        super().__init__()
        self.cross_only = cross_only
        if not self.cross_only:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

        self.activation = _get_activation_fn(activation)

        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed

    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, query, key, query_pos, key_pos, attn_mask=None):
        """
        :param query: B C Pq
        :param key: B C Pk
        :param query_pos: B Pq 3/6
        :param key_pos: B Pk 3/6
        :param value_pos: [B Pq 3/6]
        :return:
        """
        # NxCxP to PxNxC
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        if self.cross_posembed is not None:
            key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1)
        else:
            key_pos_embed = None

        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)

        if not self.cross_only:
            q = k = v = self.with_pos_embed(query, query_pos_embed)
            query2 = self.self_attn(q, k, value=v)[0]
            query = query + self.dropout1(query2)
            query = self.norm1(query)

        query2 = self.multihead_attn(query=self.with_pos_embed(query, query_pos_embed),
                                     key=self.with_pos_embed(key, key_pos_embed),
                                     value=self.with_pos_embed(key, key_pos_embed), attn_mask=attn_mask)[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        # NxCxP to PxNxC
        query = query.permute(1, 2, 0)
        return query


class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if hasattr(self, '_qkv_same_embed_dim') and self._qkv_same_embed_dim is False:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            if not hasattr(self, '_qkv_same_embed_dim'):
                warnings.warn('A new version of MultiheadAttention module has been implemented. \
                    Please re-train your model with the new module',
                              UserWarning)

            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)


def multi_head_attention_forward(query,  # type: Tensor
                                 key,  # type: Tensor
                                 value,  # type: Tensor
                                 embed_dim_to_check,  # type: int
                                 num_heads,  # type: int
                                 in_proj_weight,  # type: Tensor
                                 in_proj_bias,  # type: Tensor
                                 bias_k,  # type: Optional[Tensor]
                                 bias_v,  # type: Optional[Tensor]
                                 add_zero_attn,  # type: bool
                                 dropout_p,  # type: float
                                 out_proj_weight,  # type: Tensor
                                 out_proj_bias,  # type: Tensor
                                 training=True,  # type: bool
                                 key_padding_mask=None,  # type: Optional[Tensor]
                                 need_weights=True,  # type: bool
                                 attn_mask=None,  # type: Optional[Tensor]
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj_weight=None,  # type: Optional[Tensor]
                                 k_proj_weight=None,  # type: Optional[Tensor]
                                 v_proj_weight=None,  # type: Optional[Tensor]
                                 static_k=None,  # type: Optional[Tensor]
                                 static_v=None,  # type: Optional[Tensor]
                                 ):
    # type: (...) -> Tuple[Tensor, Optional[Tensor]]
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in differnt forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert list(query.size()) == [tgt_len, bsz, embed_dim]
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if use_separate_proj_weight is not True:
        if qkv_same:
            # self-attention
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif kv_same:
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask,
                                       torch.zeros((attn_mask.size(0), 1),
                                                   dtype=attn_mask.dtype,
                                                   device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                   dtype=key_padding_mask.dtype,
                                                   device=key_padding_mask.device)], dim=1)
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1),
                                                          dtype=attn_mask.dtype,
                                                          device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                               dtype=key_padding_mask.dtype,
                                               device=key_padding_mask.device)], dim=1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = F.softmax(
        attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None

@DETECTORS.register_module
class TransFusionDetector(BaseDetector):
    def __init__(
        self,
        first_stage_cfg,
        second_stage_modules,
        roi_head,
        image_head_cfg, 
        NMS_POST_MAXSIZE,
        num_point=1,
        freeze=False,
        **kwargs
    ):
        super(TransFusionDetector, self).__init__()
        self.single_det = builder.build_detector(first_stage_cfg, **kwargs)
        self.image_head= builder.build_detector(image_head_cfg)
        self.NMS_POST_MAXSIZE = NMS_POST_MAXSIZE
        self.num_decoder_layers = 3

        if freeze:
            print("Freeze First Stage Network")
            # we train the model in two steps 
            self.single_det = self.single_det.freeze()
            self.image_head = self.image_head.freeze()
        self.bbox_head = self.single_det.bbox_head


        self.second_stage = nn.ModuleList()
        # can be any number of modules 
        # bird eye view, cylindrical view, image, multiple timesteps, etc.. 
        for module in second_stage_modules:
            self.pc_start = module['pc_start']
            self.voxel_size = module['voxel_size']
            self.out_stride = module['out_stride']
            self.second_stage.append(builder.build_second_stage_module(module))

        self.roi_head = builder.build_roi_head(roi_head)

        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    64, 8, 128, 0.1, 'relu',
                    self_posembed=PositionEmbeddingLearned(2, 64),
                    cross_posembed=PositionEmbeddingLearned(2, 64),
                    cross_only=True,
                ))
        self.key_projection = nn.Sequential(
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,64),
            )
        self.query_projection = nn.Sequential(
            nn.Linear(512,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,64),
        )

        
        self.num_point = num_point


    def combine_loss(self, one_stage_loss, roi_loss, tb_dict):
        one_stage_loss['loss'][0] += (roi_loss)

        for i in range(len(one_stage_loss['loss'])):
            one_stage_loss['roi_reg_loss'].append(tb_dict['rcnn_loss_reg'])
            one_stage_loss['roi_cls_loss'].append(tb_dict['rcnn_loss_cls'])

        return one_stage_loss
    
    def absl_to_relative(self, absolute):
        # Going from absolute meter scale to relative pixel scale, 
        a1 = (absolute[..., 0] - self.pc_start[0]) / self.voxel_size[0] / self.out_stride 
        a2 = (absolute[..., 1] - self.pc_start[1]) / self.voxel_size[1] / self.out_stride 

        return torch.cat((a1.unsqueeze(-1), a2.unsqueeze(-1)),dim=-1)

    def relative_to_absl(self, relative):
        # Going from relative pixel scale to absolute meter scale, 
        a1 = relative[..., 0] * self.voxel_size[0] * self.out_stride  + self.pc_start[0]
        a2 = relative[..., 1] * self.voxel_size[1] * self.out_stride  + self.pc_start[1]

        return torch.cat((a1.unsqueeze(-1), a2.unsqueeze(-1)),dim=-1)

    def get_box_center(self, boxes):
        # box [List]
        centers = [] 
        for box in boxes:            
            if self.num_point == 1 or len(box['box3d_lidar']) == 0:
                centers.append(box['box3d_lidar'][:, :3])
                
            elif self.num_point == 5:
                center2d = box['box3d_lidar'][:, :2]
                height = box['box3d_lidar'][:, 2:3]
                dim2d = box['box3d_lidar'][:, 3:5]
                rotation_y = box['box3d_lidar'][:, -1]

                corners = box_torch_ops.center_to_corner_box2d(center2d, dim2d, rotation_y)

                front_middle = torch.cat([(corners[:, 0] + corners[:, 1])/2, height], dim=-1)
                back_middle = torch.cat([(corners[:, 2] + corners[:, 3])/2, height], dim=-1)
                left_middle = torch.cat([(corners[:, 0] + corners[:, 3])/2, height], dim=-1)
                right_middle = torch.cat([(corners[:, 1] + corners[:, 2])/2, height], dim=-1) 

                points = torch.cat([box['box3d_lidar'][:, :3], front_middle, back_middle, left_middle, \
                    right_middle], dim=0)

                centers.append(points)
            else:
                raise NotImplementedError()

        return centers

    def reorder_first_stage_pred_and_feature(self, first_pred, example, features):
        batch_size = len(first_pred)
        box_length = first_pred[0]['box3d_lidar'].shape[1] 
        feature_vector_length = sum([feat[0].shape[-1] for feat in features])
        
        rois = first_pred[0]['box3d_lidar'].new_zeros((batch_size, 
            self.NMS_POST_MAXSIZE, box_length 
        ))
        roi_scores = first_pred[0]['scores'].new_zeros((batch_size,
            self.NMS_POST_MAXSIZE
        ))
        roi_labels = first_pred[0]['label_preds'].new_zeros((batch_size,
            self.NMS_POST_MAXSIZE), dtype=torch.long
        )
        roi_features = features[0][0].new_zeros((batch_size, 
            self.NMS_POST_MAXSIZE, feature_vector_length 
        ))

        for i in range(batch_size):
            num_obj = features[0][i].shape[0]
            # basically move rotation to position 6, so now the box is 7 + C . C is 2 for nuscenes to
            # include velocity target

            box_preds = first_pred[i]['box3d_lidar']

            if self.roi_head.code_size == 9:
                # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y
                box_preds = box_preds[:, [0, 1, 2, 3, 4, 5, 8, 6, 7]]

            rois[i, :num_obj] = box_preds
            roi_labels[i, :num_obj] = first_pred[i]['label_preds'] + 1
            roi_scores[i, :num_obj] = first_pred[i]['scores']
            roi_features[i, :num_obj] = torch.cat([feat[i] for feat in features], dim=-1)

        example['rois'] = rois 
        example['roi_labels'] = roi_labels 
        example['roi_scores'] = roi_scores  
        example['roi_features'] = roi_features

        example['has_class_labels']= True 

        return example 

    def post_process(self, batch_dict):
        batch_size = batch_dict['batch_size']
        pred_dicts = [] 

        for index in range(batch_size):
            box_preds = batch_dict['batch_box_preds'][index]
            cls_preds = batch_dict['batch_cls_preds'][index]  # this is the predicted iou 
            label_preds = batch_dict['roi_labels'][index]

            if box_preds.shape[-1] == 9:
                # move rotation to the end (the create submission file will take elements from 0:6 and -1) 
                box_preds = box_preds[:, [0, 1, 2, 3, 4, 5, 7, 8, 6]]

            scores = torch.sqrt(torch.sigmoid(cls_preds).reshape(-1) * batch_dict['roi_scores'][index].reshape(-1)) 
            mask = (label_preds != 0).reshape(-1)

            box_preds = box_preds[mask, :]
            scores = scores[mask]
            labels = label_preds[mask]-1

            # currently don't need nms 
            pred_dict = {
                'box3d_lidar': box_preds,
                'scores': scores,
                'label_preds': labels,
                "metadata": batch_dict["metadata"][index]
            }

            pred_dicts.append(pred_dict)

        return pred_dicts 


    def forward(self, example, return_loss=True, **kwargs):

        image_out= self.image_head.forward_two_stage(example,return_loss,**kwargs)
        #image_put is a dictionary, {'results': output, 'loss': losses_merged}

        out = self.single_det.forward_with_trans_fusion(example, image_out,
            return_loss,**kwargs)

        if len(out) == 6:
            one_stage_pred, bev_feature, bev_pos, voxel_feature, final_feature, one_stage_loss = out 
            example['voxel_feature'] = voxel_feature
        elif len(out) == 3:
            one_stage_pred, bev_feature, one_stage_loss = out 
        else:
            raise NotImplementedError

        '''
        cam_feat_flatten = bev_feature.view(bev_feature.shape[0],bev_feature.shape[1],-1)[:,-64:,:].clone().detach()
        cam_feat_flatten = self.key_projection(cam_feat_flatten.permute(0,2,1)).permute(0,2,1)
        bev_pos = torch.cat([pred['bev_pos'] for pred in one_stage_pred]).to(cam_feat_flatten)
        query_pos = torch.cat([pred['query_pos'] for pred in one_stage_pred]).to(cam_feat_flatten)
        query_feat = torch.cat([pred['query_feat'] for pred in one_stage_pred]).to(cam_feat_flatten)[:,:-64,:]

        for i in range(self.num_decoder_layers):
            query_feat = self.decoder[i](query_feat, cam_feat_flatten, query_pos, bev_pos) + query_feat
        
        bev_feature = bev_feature.view(bev_feature.shape[0],bev_feature.shape[1],-1)
        query_feat = self.ffn(query_feat.permute(0,2,1)).permute(0,2,1)

        for i in range(bev_feature.shape[0]):
            bev_feature[i,-64:,one_stage_pred[i]['indices']] = query_feat[i,:,:len(one_stage_pred[i]['indices'])]
        bev_feature = bev_feature.view(bev_feature.shape[0],bev_feature.shape[1],188,188)
        '''
        
        centers_vehicle_frame = self.get_box_center(one_stage_pred)
        
        query_pos = torch.zeros(bev_feature.shape[0],2500,2).to(bev_feature)
        query_feat = torch.zeros(bev_feature.shape[0],bev_feature.shape[1]-64,2500).to(bev_feature)
        
        query_absl_to_rel = []
        for i in range(query_pos.shape[0]):
            query_pos[i,:len(centers_vehicle_frame[i]),:] = centers_vehicle_frame[i][:,:2]
            query_absl_to_rel.append(self.absl_to_relative(centers_vehicle_frame[i][:,:2]).long()%188)
            query_feat[i,:,:len(centers_vehicle_frame[i])] = bev_feature[i,:-64,query_absl_to_rel[i][:,1],query_absl_to_rel[i][:,0]]
        
        #bev_pos = torch.cat([pred['bev_pos'] for pred in one_stage_pred]).to(bev_feature)
        if bev_pos.shape[0] == 0:
            bev_pos = torch.zeros((1,2)).long()
        cam_feat_flatten = bev_feature[...,bev_pos[:,1],bev_pos[:,0]]
        cam_feat_flatten = cam_feat_flatten.view(bev_feature.shape[0],bev_feature.shape[1],-1)[:,-64:,:]
        cam_feat_flatten = self.key_projection(cam_feat_flatten.permute(0,2,1)).permute(0,2,1)
        query_feat = self.query_projection(query_feat.permute(0,2,1)).permute(0,2,1)

        bev_pos = self.relative_to_absl(bev_pos).to(bev_feature)
        bev_pos = bev_pos.unsqueeze(0)
        bev_pos = bev_pos.repeat(bev_feature.shape[0],1,1)

        for i in range(self.num_decoder_layers):
            query_feat = self.decoder[i](query_feat, cam_feat_flatten, query_pos, bev_pos) + query_feat
        #query_feat = self.ffn(query_feat.permute(0,2,1)).permute(0,2,1)

        for i in range(query_pos.shape[0]):
                bev_feature[i,-64:,query_absl_to_rel[i][:,1],query_absl_to_rel[i][:,0]] = query_feat[i,:,:len(centers_vehicle_frame[i])]
        
        # N C H W -> N H W C 
        if kwargs.get('use_final_feature', False):
            example['bev_feature'] = final_feature.permute(0, 2, 3, 1).contiguous()
        else:
            example['bev_feature'] = bev_feature.permute(0, 2, 3, 1).contiguous()


        if self.roi_head.code_size == 7 and return_loss is True:
            # drop velocity 
            example['gt_boxes_and_cls'] = example['gt_boxes_and_cls'][:, :, [0, 1, 2, 3, 4, 5, 6, -1]]

        features = [] 

        for module in self.second_stage:
            feature = module.forward(example, centers_vehicle_frame, self.num_point)
            features.append(feature)
            # feature is two level list 
            # first level is number of two stage information streams
            # second level is batch 
        
        example = self.reorder_first_stage_pred_and_feature(first_pred=one_stage_pred, example=example, features=features)

        # final classification / regression 
        batch_dict = self.roi_head(example, training=return_loss)

        if return_loss:
            roi_loss, tb_dict = self.roi_head.get_loss()

            return self.combine_loss(one_stage_loss, roi_loss, tb_dict)
        else:
            return self.post_process(batch_dict)
