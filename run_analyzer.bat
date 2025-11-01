@echo off
echo Installing required packages...
pip install -r requirements.txt

echo Starting the application...
python interrogation_analyzer_gui.py
pause 