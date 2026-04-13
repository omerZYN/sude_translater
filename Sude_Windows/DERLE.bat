@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ==========================================
echo    SUDE - HotWire Foam Cutter
echo    Windows Build Script
echo ==========================================
echo.
echo Calisma klasoru: %CD%
echo.

echo [1/3] Gerekli kutuphaneler yukleniyor...
pip install ezdxf numpy matplotlib pyinstaller
if %errorlevel% neq 0 (
    echo HATA: pip install basarisiz. Python kurulu mu?
    pause
    exit /b 1
)

echo.
echo [2/3] Sude.exe derleniyor...
pyinstaller Sude.spec --clean --noconfirm
if %errorlevel% neq 0 (
    echo HATA: Derleme basarisiz!
    pause
    exit /b 1
)

echo.
echo [3/3] Tamamlandi!
echo.
echo ==========================================
echo  Sude.exe dosyasi: dist\Sude.exe
echo ==========================================
echo.

explorer dist
pause
