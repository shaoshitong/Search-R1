"""Model equations copied from warmup 7a8ff6a, without changing state keys."""
import torch
from torch import nn


class PackedVarianceHead(nn.Module):
    def __init__(self, hidden=3584, dimension=256, max_positions=2048):
        super().__init__()
        self.projection = nn.Linear(hidden, dimension)
        self.types = nn.Embedding(3, dimension)
        self.positions = nn.Embedding(max_positions, dimension)
        self.cls = nn.Parameter(torch.empty(1, 1, dimension))
        nn.init.normal_(self.cls, std=.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dimension, 8, 4 * dimension, dropout=0.,
                activation='gelu', batch_first=True, norm_first=True) for _ in range(2)
        ])
        self.norm = nn.LayerNorm(dimension)
        self.output = nn.Linear(dimension, 1)

    def forward(self, hidden, mask, positions, types):
        x = (self.projection(hidden.masked_fill(~mask[..., None], 0))
             + self.types(types) + self.positions(positions))
        x = torch.cat((self.cls.expand(len(x), -1, -1), x), 1)
        valid = torch.cat((torch.ones((len(mask), 1), dtype=torch.bool,
                                     device=mask.device), mask), 1)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=~valid)
        return self.output(self.norm(x[:, 0])).squeeze(-1).float()


class PackedVarianceProbe(nn.Module):
    def __init__(self, block, rotary, cache_position, head):
        super().__init__()
        self.block = block
        self.rotary = rotary
        self.cache_position = cache_position
        self.group_head = head

    def forward(self, batch):
        hidden = batch['features'].detach()
        mask, positions = batch['attention_mask'], batch['position_ids']
        kwargs = dict(attention_mask=mask, position_ids=positions,
            position_embeddings=self.rotary(hidden, positions),
            use_cache=False, output_attentions=False)
        if self.cache_position:
            kwargs['cache_position'] = torch.arange(hidden.shape[1], device=hidden.device)
        value = self.block(hidden, **kwargs)
        value = value[0] if isinstance(value, tuple) else value
        return self.group_head(value, mask, positions, batch['token_types'])
