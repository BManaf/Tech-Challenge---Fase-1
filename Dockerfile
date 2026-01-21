# Imagem base com Python 3.10
FROM python:3.10-slim

# Configurações para evitar arquivos .pyc e melhorar logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependências de sistema necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Diretório de trabalho
WORKDIR /app

# Copia requirements primeiro (melhor cache)
COPY requirements.txt /app/requirements.txt

# Instala dependências Python
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Copia o restante do projeto
COPY . /app

# Comando padrão: desafio principal
CMD ["python", "src/main_tabular.py"]
