import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel

# casing features not present in uncased encoder models
N_SHAPES = 5


class BilinearMatrixAttention(nn.Module):
    """
    Arch biaffine with EfficientSDP's normalization trick.
    Norm: sqrt((d1+ d2)/2) is applied to every forward pass
    """

    def __init__(
        self, matrix_1_dim, matrix_2_dim, use_input_biases=True, arc_norm=True
    ):
        super().__init__()

        if use_input_biases:
            matrix_1_dim += 1
            matrix_2_dim += 1

        self.use_input_biases = use_input_biases
        self._weight_matrix = nn.Parameter(torch.empty(matrix_1_dim, matrix_2_dim))
        nn.init.xavier_uniform(self._weight_matrix)
        self.arc_norm = arc_norm
        self.scale_norm = (
            math.sqrt((matrix_1_dim + matrix_2_dim) / 2) if arc_norm else 1.0
        )

    def forward(self, matrix_1, matrix_2):
        if self.use_input_biases:
            matrix_1 = torch.cat([matrix_1, torch.ones_like(matrix_1[..., :1])], dim=-1)
            matrix_2 = torch.cat([matrix_2, torch.ones_like(matrix_2[..., :1])], dim=-1)
        final = matrix_1 @ self._weight_matrix @ matrix_2.transpose(-1, -2)
        return self.normalize(final)

    def normalize(self, adj):
        return adj / self.scale_norm if self.arc_norm else adj


# bert-mini wrapper with sub-word dropping. (uses the first-subword vectors)
class Encoder(nn.Module):
    def __init__(
        self,
        encoder_name,
        n_pos,
        n_edge_labels,
        shape_dim=32,
        arc_representation_dim=384,
        tag_representation_dim=128,
        dropout=0.33,
        encoder_dropout=0.1,
    ):
        super().__init__()
        cfg = AutoConfig.from_pretrained(encoder_name)
        cfg.hidden_dropout_prob = encoder_dropout
        cfg.attention_probs_dropout_prob = encoder_dropout
        self.encoder = AutoModel.from_pretrained(encoder_name, config=cfg)
        h = cfg.hidden_size
        self.shape_emb = nn.Embedding(N_SHAPES, shape_dim)
        encoder_dim = h + shape_dim
        # learned ROOT representation vector, prepended at positon 0
        self._head_sentinel = nn.Parameter(torch.randn(encoder_dim))
        self._dropout = nn.Dropout(dropout)

        # POS output head
        self.pos_head = nn.Linear(encoder_dim, n_pos)
        nn.init.zeros_(self.pos_head.bias)

        # arc projections + normalized biaffine
        self.head_arc_feedforward = nn.Linear(encoder_dim, arc_representation_dim)
        self.dep_arc_feedforward = nn.Linear(encoder_dim, arc_representation_dim)

        self.arc_bilinear = BilinearMatrixAttention(
            arc_representation_dim,
            arc_representation_dim,
            use_input_biases=True,
            arc_norm=True,
        )

        # label projections (head_tag/dep_tag feedforwards)
        self.head_tag_feedforward = nn.Linear(encoder_dim, tag_representation_dim)
        self.dep_tag_feedforward = nn.Linear(encoder_dim, tag_representation_dim)

    def forward(self, input_ids, attention_mask, word_index, shapes):
        """
        input_ids [B, T] subword ids
        attention_mask [B, T]
        word_index [B, W+1] subword position of ROOT(0) + first subword of each word.
        shapes [B, W]
        """
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        # gather first-subword vectors to get [B, W+1, H]
        idx = word_index.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        encoded_text_input = hidden.gather(1, idx)

        # ROOT row position 0 gets zeros
        sh = self.shape_emb(shapes)  # [B, W, shape_dim]
        zero = sh.new_zeros(sh.size(0), 1, sh.size(-1))  # [B, W+1, shape_dim]
        sh = torch.cat([zero, sh], dim=1)
        encoded_text_input = torch.cat(
            [encoded_text_input, sh], dim=-1
        )  # [B, W+1, encoder_dim]

        # replace position 0 with learned ROOT
        batch_size = encoded_text_input.size(0)
        head_sentinel = self._head_sentinel.view(1, 1, -1).expand(batch_size, 1, -1)
        encoded_text_input = torch.cat(
            [head_sentinel, encoded_text_input[:, 1:, :]], dim=1
        )
        encoded_text_input = self._dropout(encoded_text_input)

        pos_logits = self.pos_head(encoded_text_input[:, 1:, :])

        head_arc = self._dropout(F.elu(self.head_arc_feedforward(encoded_text_input)))
        dep_arc = self._dropout(F.elu(self.dep_arc_feedforward(encoded_text_input)))
        attended_arcs = self.arc_bilinear(head_arc, dep_arc)  # [B, W+1, W+1]

        # label vectors
        head_tag = self._dropout(F.elu(self.head_tag_feedforward(encoded_text_input)))
        dep_tag = self._dropout(F.elu(self.dep_tag_feedforward(encoded_text_input)))

        return pos_logits, attended_arcs, head_tag, dep_tag


def get_head_tags(head_tag, dept_tag, head_indices, tag_bilinear):
    """
    Score edge-label logits for each word against its head.

    Call with GOLD heads during loss, PREDICTED heads during decode — same
    function, and that dual use is the entire train/decode asymmetry.

    Args:
        head_tag     : [B, seq, tag_dim]   each token's "as-head" label vector
        dept_tag     : [B, seq, tag_dim]   each token's "as-dependent" label vector
        head_indices : [B, seq]            head index per word; 0 = ROOT
        tag_bilinear : nn.Bilinear(tag_dim, tag_dim, n_labels)

    Returns:
        head_tag_logits : [B, seq, n_labels]
    """
    B = head_tag.size(0)
    batch_range = torch.arange(B, device=head_tag.device).unsqueeze(1)   # [B, 1]

    # Guard: head_indices is used to index the seq axis, so it must be in range.
    # Gold heads always are; predicted heads may be dirty, so clamp defensively.
    head_indices = head_indices.clamp(0, head_tag.size(1) - 1)

    # Advanced indexing: selected_head[b, i] = head_tag[b, head_indices[b, i]]
    # i.e. for dependent i, fetch the "as-head" vector of the token it attaches to.
    selected_head = head_tag[batch_range, head_indices].contiguous()      # [B, seq, tag_dim]

    return tag_bilinear(selected_head, dept_tag)                          # [B, seq, n_labels]


def construct_loss(attended_arcs, head_tag_logits, head_indices, head_tags, mask):
    """
    Arc + label negative log-likelihood, mean over valid (non-pad, non-ROOT) words.

    Args:
        attended_arcs   : [B, seq, seq]      RAW arc logits; dim 2 = candidate heads
        head_tag_logits : [B, seq, n_labels] output of get_head_tags with GOLD heads
        head_indices    : [B, seq]           gold head per word, 0 = ROOT
        head_tags       : [B, seq]           gold label per word
        mask            : [B, seq]           1 for real tokens (incl. ROOT), 0 pad

    Returns:
        arc_nll, tag_nll : two scalars
    """
    # Ignore padding AND the ROOT sentinel at position 0. cross_entropy skips any
    # position whose target == -100, and reduction="mean" then averages over
    # exactly the non-ignored positions — which equals their (mask.sum() - B)
    # denominator by construction.
    ignore = ~mask.bool()
    ignore[:, 0] = True

    arc_targets = head_indices.masked_fill(ignore, -100)
    tag_targets = head_tags.masked_fill(ignore, -100)

    B, S, _ = attended_arcs.shape
    arc_nll = F.cross_entropy(
        attended_arcs.reshape(B * S, S),                 # each candidate head is a class
        arc_targets.reshape(B * S),
        ignore_index=-100,
        reduction="mean",
    )
    tag_nll = F.cross_entropy(
        head_tag_logits.reshape(B * S, head_tag_logits.size(-1)),
        tag_targets.reshape(B * S),
        ignore_index=-100,
        reduction="mean",
    )
    return arc_nll, tag_nll
