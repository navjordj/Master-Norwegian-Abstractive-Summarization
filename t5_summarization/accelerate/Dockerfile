FROM huggingface/accelerate-gpu

COPY requirements.txt /tmp
WORKDIR /tmp

RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

RUN source activate accelerate && \
    python3 -m pip install -r requirements.txt 

WORKDIR /

RUN echo "source activate accelerate" >> ~/.profile

# Activate the virtualenv
CMD ["/bin/bash"]
