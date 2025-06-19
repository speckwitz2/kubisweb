FROM python:3.12

# Declaring absolute directory path for the system and copy all
WORKDIR /app
COPY . .

RUN touch /app/database/logs.db

# Install depedencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN pip install -U torch torchvision -f https://download.pytorch.org/wh1/cu101/torch_stable.html
RUN pip install cython pyyaml
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' --no-build-isolation
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
RUN pip install -r requirements.txt --no-build-isolation

# Declaring custom volume
VOLUME "/app/model/weights"

# EXPOSE service port
EXPOSE 5000

CMD [ "python", "main.py" ]