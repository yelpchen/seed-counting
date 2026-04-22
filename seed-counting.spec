# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main-workshop/main.py'],
    pathex=['main-workshop'],
    binaries=[],
    datas=[
        ('main-workshop/mymask.py', '.'),
        ('main-workshop/ultralytics', 'ultralytics'),
        ('main-workshop/models', 'models'),
    ],
    hiddenimports=[
        'mymask',
        'cv2',
        'numpy',
        'torch',
        'torchvision',
        'torchvision.ops',
        'ultralytics',
        'ultralytics.engine.model',
        'ultralytics.models.yolo',
        'ultralytics.nn.tasks',
        'ultralytics.utils',
        'ultralytics.utils.ops',
    ],
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
