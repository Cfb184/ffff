FROM python:3.10-slim
WORKDIR /app
COPY RMB30026.py /app/

# 配置清华源加速
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 先安装基础库
RUN pip install --no-cache-dir pandas numpy flask flask-cors

# 专门安装极其轻量的 CPU 版 PyTorch，避免镜像体积暴增
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

EXPOSE 5026
CMD ["python", "RMB30026.py"]