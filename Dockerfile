# Dockerfile

# 1. Start with an official Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file into the container
COPY requirements.txt .

# 4. Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your entire project code into the container
COPY . .

# 6. Expose the port the app will run on
EXPOSE 5000

# 7. Define the command to run your application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5000"]