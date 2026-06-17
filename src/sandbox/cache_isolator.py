"""
Layer 1: File System Isolation.
From: triton_competition_anti_cheat_guide.md - Section 3
"""
import os
import shutil
import tempfile


class CacheIsolator:
    """缓存目录隔离器"""

    CACHE_PATHS = [
        '~/.triton',
        '~/.cache/triton',
        '~/.torch',
        '~/.cache/torch',
        '~/.nv',
    ]

    def __init__(self):
        self.original_home = os.environ.get('HOME', '/root')
        self.isolated_home = None

    def isolate(self):
        """创建隔离环境"""
        # 创建隔离HOME
        self.isolated_home = tempfile.mkdtemp(prefix='isolated_home_')
        os.environ['HOME'] = self.isolated_home

        # 设置所有缓存变量指向临时目录或禁用
        os.environ['TRITON_CACHE_DIR'] = os.path.join(self.isolated_home, '.triton')
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(self.isolated_home, '.torch')
        os.environ['CUDA_CACHE_DISABLE'] = '1'
        os.environ['XDG_CACHE_HOME'] = os.path.join(self.isolated_home, '.cache')

        # 创建必要的目录结构
        os.makedirs(os.environ['TRITON_CACHE_DIR'], exist_ok=True)
        os.makedirs(os.environ['TORCHINDUCTOR_CACHE_DIR'], exist_ok=True)

        return self.isolated_home

    def cleanup(self):
        """清理隔离环境"""
        if self.isolated_home and os.path.exists(self.isolated_home):
            shutil.rmtree(self.isolated_home, ignore_errors=True)
        os.environ['HOME'] = self.original_home

    def __enter__(self):
        return self.isolate()

    def __exit__(self, *args):
        self.cleanup()
