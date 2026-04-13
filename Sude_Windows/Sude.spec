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
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=False,
    icon=None,
)
