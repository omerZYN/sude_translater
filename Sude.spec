# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['hotwire_cutter.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['ezdxf', 'numpy', 'matplotlib', 'matplotlib.backends.backend_tkagg', 'mpl_toolkits.mplot3d', 'PIL', 'PIL._tkinter_finder'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['scipy', 'pandas'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Sude',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

app = BUNDLE(
    exe,
    name='Sude.app',
    icon=None,
    bundle_identifier='com.sude.hotwire',
    info_plist={
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleName': 'Sude',
        'NSHighResolutionCapable': True,
    },
)
