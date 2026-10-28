"""DINOv2 backbone contracts; new training uses L, legacy checkpoints use B."""
BACKBONE_DIMS = {'dinov2_vits14': 384, 'dinov2_vitb14': 768,
                 'dinov2_vitl14': 1024, 'dinov2_vitg14': 1536}
DEFAULT_BACKBONE = 'dinov2_vitl14'
LEGACY_BACKBONE = 'dinov2_vitb14'


def feature_width(backbone, raw_dim=None):
    if backbone not in BACKBONE_DIMS:
        raise ValueError(f'Unsupported DINOv2 backbone: {backbone}')
    expected = BACKBONE_DIMS[backbone]
    if raw_dim is not None and raw_dim != expected:
        raise ValueError(f'{backbone} produces {expected}-D features, not {raw_dim}')
    return expected


def cache_spec(metadata):
    """Metadata-less v2 backbone fields denote historical ViT-B/14 caches."""
    backbone = metadata.get('backbone', LEGACY_BACKBONE)
    width = feature_width(backbone, metadata.get('feature_dim'))
    if metadata.get('tokens_per_camera', 257) != 257:
        raise ValueError('Expected 256 patches plus CLS at 224x224')
    return backbone, width


def validate_cache(metadata, backbone, raw_dim):
    cached = cache_spec(metadata)
    if cached != (backbone, feature_width(backbone, raw_dim)):
        raise ValueError(f'Cache {cached} does not match requested vision backbone {(backbone, raw_dim)}; generate a separate feature cache')


def add_vision_arguments(parser):
    parser.add_argument('--vision-backbone', choices=tuple(BACKBONE_DIMS), default=DEFAULT_BACKBONE)
    parser.add_argument('--raw-dim', type=int, default=None,
                        help='Visual feature width; inferred from backbone and checked if supplied')
