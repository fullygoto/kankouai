"""Compatibility shim for legacy `watermark_utils`.
This module forwards calls to `watermark_ext` with lazy imports
to avoid circular-import during module import time.
Generated from names imported by watermark_ext.
"""

def DEFAULT_OPACITY(*args, **kwargs):
    from watermark_ext import DEFAULT_OPACITY as _impl
    return _impl(*args, **kwargs)

def DEFAULT_SCALE(*args, **kwargs):
    from watermark_ext import DEFAULT_SCALE as _impl
    return _impl(*args, **kwargs)

def DEFAULT_WATERMARK_SUFFIX(*args, **kwargs):
    from watermark_ext import DEFAULT_WATERMARK_SUFFIX as _impl
    return _impl(*args, **kwargs)

def WatermarkOptions(*args, **kwargs):
    from watermark_ext import WatermarkOptions as _impl
    return _impl(*args, **kwargs)

def apply_watermark(*args, **kwargs):
    from watermark_ext import apply_watermark as _impl
    return _impl(*args, **kwargs)

def atomic_write(*args, **kwargs):
    from watermark_ext import atomic_write as _impl
    return _impl(*args, **kwargs)

def choose_unique_filename(*args, **kwargs):
    from watermark_ext import choose_unique_filename as _impl
    return _impl(*args, **kwargs)

def derivative_path(*args, **kwargs):
    from watermark_ext import derivative_path as _impl
    return _impl(*args, **kwargs)

def ensure_within_media(*args, **kwargs):
    from watermark_ext import ensure_within_media as _impl
    return _impl(*args, **kwargs)

def list_media_files(*args, **kwargs):
    from watermark_ext import list_media_files as _impl
    return _impl(*args, **kwargs)

def list_watermark_files(*args, **kwargs):
    from watermark_ext import list_watermark_files as _impl
    return _impl(*args, **kwargs)

def max_batch_size(*args, **kwargs):
    from watermark_ext import max_batch_size as _impl
    return _impl(*args, **kwargs)

def media_path_for(*args, **kwargs):
    from watermark_ext import media_path_for as _impl
    return _impl(*args, **kwargs)

def strip_derivative_suffix(*args, **kwargs):
    from watermark_ext import strip_derivative_suffix as _impl
    return _impl(*args, **kwargs)

def validate_upload(*args, **kwargs):
    from watermark_ext import validate_upload as _impl
    return _impl(*args, **kwargs)

__all__ = [
    'DEFAULT_OPACITY',
    'DEFAULT_SCALE',
    'DEFAULT_WATERMARK_SUFFIX',
    'WatermarkOptions',
    'apply_watermark',
    'atomic_write',
    'choose_unique_filename',
    'derivative_path',
    'ensure_within_media',
    'list_media_files',
    'list_watermark_files',
    'max_batch_size',
    'media_path_for',
    'strip_derivative_suffix',
    'validate_upload',
]
