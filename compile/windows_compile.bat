cd ..
CALL .\venv\Scripts\activate.bat
cd compile\windows
python -OO -m PyInstaller --clean -w --onefile --upx-dir ..\upx-3.96-win64 --name IVP_ODE_solver -p ..\..\src -p ..\..\src\equations ..\..\src\main.py