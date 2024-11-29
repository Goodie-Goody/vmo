# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.11.9-slim

# Create a working directory.
WORKDIR /app

# Copy the current directory contents into the container at /app.
COPY . /app

# Install the required packages.
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 to the outside world.
EXPOSE 8080

# Run the app using gunicorn.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "vmo:app"]