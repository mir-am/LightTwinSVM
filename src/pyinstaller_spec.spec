# -*- mode: python -*-
# LightTwinSVM Program - Simple and Fast
# Version: 0.2.0-alpha - 2018-05-30
# Developer: Mir, A. (mir-am@hotmail.com)
# License: GNU General Public License v3.0

# PyInstaller specification file for generating pre-built Windows binary.

block_cipher = None

# 'scipy._lib.messagestream', 'pandas._libs.tslibs.timedeltas', 'pandas._libs.tslibs.np_datetime'
#'C:\\Users\\Mir\\mirenv\\Lib\\site-packages\\scipy\\extra-dll'
a = Analysis(['main.py'],
             pathex=[],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='LightTwinSVM',
          debug=True,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
