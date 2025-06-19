FROM python:3.12-slim

# Declaring absolute directory path for the system and copy all
WORKDIR /app
COPY . .

RUN mkdir -p /app/database && \
    touch /app/database/logs.db

# Install depedencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get install -y git
RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install -U torch torchvision --no-cache-dir
RUN pip install cython pyyaml --no-cache-dir
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation --no-cache-dir
RUN pip install -r requirements.txt --no-build-isolation

# Cleaning up
RUN rm -rf ~/.cache/pip
RUN apt-get purge -y libgl1-mesa-glx git
RUN pip uninstall -y cython pyyaml
RUN apt-get purge -y build-essential

RUN apt-get autoremove -y 
RUN rm -rf /var/lib/apt/lists/*

# Declaring custom volume
VOLUME "/app/model/weights"

# EXPOSE service port
EXPOSE 5000

CMD [ "python", "main.py" ]