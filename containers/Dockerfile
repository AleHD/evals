WORKDIR /workspace

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.01-py3
ARG TRANSFORMERS_BRANCH=main
ARG HARNESS_BRANCH=granular_eval
ARG MEGATRON_BRANCH=main

FROM ${BASE_IMAGE}

# Install dependencies.
RUN git clone https://github.com/swiss-ai/transformers.git && \
    cd transformers && \
    git checkout $TRANSFORMERS_BRANCH && \
    pip install -e . && \
    cd ..

RUN git clone https://github.com/swiss-ai/lm-evaluation-harness.git && \
    cd lm-evaluation-harness && \
    git checkout $HARNESS_BRANCH && \
    pip install -e . && \
    cd ..
