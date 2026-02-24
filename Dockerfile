FROM python:3.13-slim

# Install FFmpeg, fonts, and wget
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg fonts-liberation wget unzip fontconfig && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download Poppins font from Google Fonts
RUN mkdir -p /usr/share/fonts/poppins && \
    wget -q "https://fonts.google.com/download?family=Poppins" -O /tmp/poppins.zip && \
    unzip -o /tmp/poppins.zip -d /usr/share/fonts/poppins && \
    rm /tmp/poppins.zip && \
    fc-cache -f -v

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY server.py .

# Expose port
EXPOSE 8000

# Run the server
CMD ["python", "server.py"]
