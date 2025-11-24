# Usa uma imagem oficial do Python 3.12 como base
FROM python:3.12-slim

# Define o diretório de trabalho
WORKDIR /usr/src/app

# Copia e instala as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do seu código para o container
COPY . .

# Define o comando de inicialização Gunicorn
CMD ["gunicorn", "app:app"]
