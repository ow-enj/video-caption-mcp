FROM python:3.13-slim

# Install FFmpeg and font tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg wget fontconfig && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download Poppins font from GitHub (reliable source)
RUN mkdir -p /usr/share/fonts/poppins && \
    wget -q "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Bold.ttf" -O /usr/share/fonts/poppins/Poppins-Bold.ttf && \
    wget -q "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Regular.ttf" -O /usr/share/fonts/poppins/Poppins-Regular.ttf && \
    wget -q "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-SemiBold.ttf" -O /usr/share/fonts/poppins/Poppins-SemiBold.ttf && \
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
