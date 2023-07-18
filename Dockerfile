FROM python:3.9

WORKDIR /app

# Copia el archivo requirements.txt e instala las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia los archivos necesarios para tu aplicación
COPY app/ .


# Configura el comando para ejecutar la aplicación Flask
CMD ["python", "main.py"]
