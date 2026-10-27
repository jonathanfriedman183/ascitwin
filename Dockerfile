FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace

COPY pyproject.toml requirements.txt README.md ./
COPY ascitwin ./ascitwin
COPY configs ./configs

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -e ".[export]"

ENTRYPOINT ["ascitwin"]
CMD ["--help"]
