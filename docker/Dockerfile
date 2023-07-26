FROM twosixarmory/pytorch
MAINTAINER gozum
RUN /opt/conda/bin/pip install einops
RUN /opt/conda/bin/pip install ema_pytorch
RUN /opt/conda/bin/pip install accelerate
RUN /opt/conda/bin/pip install pytorch-fid
RUN /opt/conda/bin/pip install ipdb
RUN /opt/conda/bin/pip install torch --upgrade
RUN /opt/conda/bin/pip install torchvision --upgrade
RUN /opt/conda/bin/pip install git+https://github.com/gabrielgozum/adversarial-robustness-toolbox-pytorch2_0.git
# RUN /opt/conda/bin/pip install openmim --no-cache-dir
# RUN mim install mmdet
CMD tail -f /dev/null