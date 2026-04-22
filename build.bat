@echo off
pip install pyinstaller
pyinstaller seed-counting.spec --clean
pause
