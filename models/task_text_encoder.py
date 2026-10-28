"""Pretrained, open-vocabulary task text with frozen contextual features."""
from collections import OrderedDict
import torch
from torch import nn

DEFAULT_TEXT_MODEL = 'distilbert/distilbert-base-uncased'


class DistilBertTaskEncoder(nn.Module):
    """Frozen DistilBERT CLS features -> trainable policy projection.

    Cache only detached raw features on CPU, never the trainable projection.
    Pretrained backbone weights remain in state_dict for checkpoint fidelity.
    Tokenizer assets are resolved through the saved model name/revision (or a
    local pretrained directory), using the normal Hugging Face cache.
    """
    def __init__(self, dim, model_name=DEFAULT_TEXT_MODEL, max_length=128,
                 revision='main', cache_size=1024):
        super().__init__()
        if dim < 1 or max_length < 2 or cache_size < 0:
            raise ValueError('Text dimension/max length must be positive; cache size must be nonnegative')
        from transformers import AutoTokenizer, AutoModel
        self.model_name, self.revision = model_name, revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.backbone = AutoModel.from_pretrained(model_name, revision=revision)
        if self.backbone.config.model_type != 'distilbert':
            raise ValueError('Task text requires a DistilBERT checkpoint')
        if not 2 <= max_length <= self.backbone.config.max_position_embeddings:
            raise ValueError('Text max length must fit DistilBERT positional embeddings')
        self.backbone.float().requires_grad_(False).eval()
        self.projection = nn.Linear(self.backbone.config.dim, dim)
        self.max_length = max_length
        self.cache_size = cache_size
        self._features = OrderedDict()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        self._features.clear()
        return super()._load_from_state_dict(*args, **kwargs)

    def forward(self, texts):
        if not texts or any(not isinstance(text, str) for text in texts):
            raise ValueError('Expected a nonempty batch of task instruction strings')
        device = self.projection.weight.device
        missing = list(dict.fromkeys(t for t in texts if t not in self._features))
        fresh = {}
        if missing:
            batch = self.tokenizer(missing, padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors='pt')
            batch = {key:value.to(device) for key,value in batch.items()}
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                features = self.backbone(**batch).last_hidden_state[:,0].float().cpu()
            fresh = dict(zip(missing, features.unbind(0)))
        # Resolve this batch before eviction, even when it exceeds cache size.
        raw = torch.stack([fresh[t] if t in fresh else self._features[t] for t in texts])
        self._features.update(fresh)
        for text in texts:
            if text in self._features:
                self._features.move_to_end(text)
        while len(self._features) > self.cache_size:
            self._features.popitem(last=False)
        return self.projection(raw.to(device=device, dtype=self.projection.weight.dtype))


def add_task_text_arguments(parser, allow_bag=True):
    parser.add_argument('--text-encoder-type', choices=('distilbert','bag') if allow_bag else ('distilbert','clip'),
                        default='distilbert')
    parser.add_argument('--text-model-name', default=DEFAULT_TEXT_MODEL,
                        help='Pretrained DistilBERT model ID or local pretrained directory')
    parser.add_argument('--text-model-revision', default='main', help='Hugging Face revision or commit')
    parser.add_argument('--text-max-length', type=int, default=128,
                        help='Maximum tokenizer sequence length, including special tokens')
