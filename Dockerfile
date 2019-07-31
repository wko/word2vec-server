FROM python:3

RUN pip install --upgrade pip && \
    pip install flask flask-restful gensim && \
    mkdir -p /root/myapp


COPY ./w2v_server.py /root/myapp/w2v_server.py

RUN mkdir -p /root/data


VOLUME ["/root/data"]
WORKDIR /root/myapp
CMD python w2v_server.py --model $MODEL_PATH
