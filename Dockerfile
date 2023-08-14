FROM python:3.7.17-slim

ADD ./Code/counterfactual_script.py .

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY ./ ./

CMD ["python", "./Code/counterfactual_script.py"]