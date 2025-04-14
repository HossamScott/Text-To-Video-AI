FROM python:3.12

WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    imagemagick \
    libmagickwand-dev \
    ghostscript \
    fonts-dejavu \
    fonts-freefont-ttf \
    && rm -rf /var/lib/apt/lists/*

# Fix ImageMagick policies
# Configure ImageMagick policy
RUN mv /etc/ImageMagick-6/policy.xml /etc/ImageMagick-6/policy.xml.bak && \
    echo '<policymap>' > /etc/ImageMagick-6/policy.xml && \
    echo '<policy domain="coder" rights="read|write" pattern="*" />' >> /etc/ImageMagick-6/policy.xml && \
    echo '<policy domain="path" rights="read|write" pattern="@*" />' >> /etc/ImageMagick-6/policy.xml && \
    echo '</policymap>' >> /etc/ImageMagick-6/policy.xml

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt && pip install opencv-python-headless

# Copy the entire project structure
COPY . .

# 6. Setup temp directory
RUN mkdir -p /tmp/moviepy && \
    chmod 777 /tmp && \
    chmod 777 /tmp/moviepy
    
# Set environment variables (Replace with actual API keys)
ENV PEXELS_KEY="aXA4IlmjYKdzM9R7JZX6l4SwVmxTsaJbMvp9l7jf7rE9VVbh5lbxvoKn"
ENV OPENROUTER_API_KEY="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"

ENV FLASK_ENV=production
ENV TEMP=/tmp

# Expose the port the app runs on
EXPOSE 5050

# Run the application
CMD ["python", "app.py"]