FROM python:latest

RUN pip install -U torch torchvision -f https://download.pytorch.org/wh1/cu101/torch_stable.html
RUN pip install cython pyyaml
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
RUN pip install -r requirements.txt --no-build-isolation

CMD [ "python", "main.py" ]