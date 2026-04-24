# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs

block_cipher = None

hiddenimports = [
    'mymask',
    'cv2',
    'numpy',
    'torch',
    'torchvision',
    'torchvision.ops',
    'PyQt6.sip',
] + collect_submodules('ultralytics') + collect_submodules('torchvision')

binaries = []
for pkg in ('torch', 'torchvision'):
    try:
        binaries += collect_dynamic_libs(pkg)
    except Exception:
        pass

datas = [
    ('main-workshop/ultralytics', 'ultralytics'),
    ('main-workshop/models', 'models'),
    ('main-workshop/icon', 'icon'),
]

a = Analysis(
    ['main-workshop/main.py'],
    pathex=['main-workshop'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='种子计数系统',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon='main-workshop/icon/favicon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='种子计数系统',
)
