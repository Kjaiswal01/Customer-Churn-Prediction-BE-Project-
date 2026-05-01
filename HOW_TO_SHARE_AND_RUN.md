# Customer Churn Prediction - Share and Run Guide

## Problem ka reason

`ModuleNotFoundError: No module named 'sqlalchemy'` ka matlab code kharab nahi hai. Iska matlab doosre laptop me required Python package install nahi hua.

Python projects me packages folder ke saath share nahi kiye jaate. Isliye ZIP receive karne wale person ko project ke dependencies install karni padti hain. Is project me ye kaam automatic karne ke liye `setup_and_run.bat` aur `RUN_PROJECT.bat` ready hai.

Extra safety ke liye `app.py` me dependency bootstrap bhi add hai. Agar koi galti se direct `streamlit run app.py` chala de aur `sqlalchemy` jaise packages missing hon, app same Python environment me `requirements.txt` install karne ki koshish karega.

## Doosre laptop par run karne ka easiest method

1. ZIP extract karo.
2. Extracted folder open karo.
3. `RUN_PROJECT.bat` par double-click karo.
4. Pehli baar internet required hoga, kyunki packages install honge.
5. Browser me ye URL open hoga:

```text
http://localhost:8501
```

## Agar CMD se run karna ho

```bat
cd customer-churn-prediction
RUN_PROJECT.bat
```

## Required software

Install hona chahiye:

- Python 3.11 or Python 3.12
- Internet connection for first setup

Python install karte time `Add python.exe to PATH` tick karna zaroori hai.

## Important files

- `requirements.txt`: saare required packages ki list, including `sqlalchemy`
- `setup_and_run.bat`: virtual environment banata hai, packages install karta hai, app run karta hai
- `RUN_PROJECT.bat`: one-click start file
- `.gitignore`: `.venv`, cache, logs, local DB files ko share/package se bahar rakhne ke liye

## ZIP banate time kya include/exclude karna hai

Include:

- Python files: `app.py`, `enterprise_*.py`, `data_pipeline.py`, etc.
- `requirements.txt`
- `setup_and_run.bat`
- `RUN_PROJECT.bat`
- `data/`
- `models/`
- `templates/`
- `static/`

Exclude:

- `.venv/`
- `__pycache__/`
- `.git/`
- `.env`
- `*.log`

## Agar phir bhi error aaye

Folder ke andar CMD open karke ye run karo:

```bat
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Most common fix yehi hai. Missing module error ka permanent solution hai ki hamesha `requirements.txt` ke through dependencies install karo, direct `python app.py` ya global Python se app run na karo.

Important: hamesha latest updated folder ka ZIP bhejo. Agar purana ZIP WhatsApp se already bheja hua hai, usme ye fixes nahi honge.
