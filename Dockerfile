FROM hendrixroa/python-ta-lib:latest
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip 
RUN pip --no-cache-dir install -r /app/requirements.txt
ENTRYPOINT ["python"]
CMD ["docker_deploy_test.py"]
