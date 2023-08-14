# U-Heal
Private code repository for the U-Heal project

In able to run the CARLA experiments you need to follow the next steps:
1. install any version of Python 3.7
2. Ceate a virtual environment and install all requirements:
    - python -m venv .venv
    - pip install -r requirements.txt
3. Execute the examples described in the counterfactual_script.py file

## Docker
1. cd C:\Program Files (x86)\Xming
2. Xming.exe -ac
3. docker run -it --rm -e DISPLAY=192.168.68.105:0.0 --network="host" --name gui_container python-u-heal