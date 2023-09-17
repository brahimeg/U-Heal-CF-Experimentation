# U-Heal
Code repository for the U-Heal project

In able to run the CARLA experiments using your environment you need to follow the next steps:
1. install any version of Python 3.7
2. Ceate a virtual environment and install all requirements:
    - python -m venv .venv
    - pip install -r requirements.txt
3. Execute the examples in the Code/counterfactual_script.py file or use Docker image to run the UI directly

# Docker setup:
1. Install docker engine from https://docs.docker.com/engine/install/
2. Download the docker image from your secure link
3. Open terminal and run the following command: 
   - docker load -i \<path to the image tar file\>
4. Run the docker image with the following command:
   - docker run --publish 5000:5000 \<image name\>
5. Open your browser and go to http://localhost:5000/ to see the UI
