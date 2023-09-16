# FROM python:3.7.17-slim

# ADD ./Code/counterfactual_script_UI.py .

# COPY ./requirements.txt ./
# RUN apt-get install -y python3-tk
# # RUN pip install --no-cache-dir -r requirements.txt
# COPY ./ ./

# CMD ["python", "./Code/counterfactual_script_UI.py"]

FROM python:3.7.17-slim

COPY ./requirements.txt ./
RUN apt-get update -y
RUN apt-get install tk -y
RUN pip install --no-cache-dir -r requirements.txt

COPY ./ ./
EXPOSE 5000
ENV FLASK_APP=./Code/UI/app.py
CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]