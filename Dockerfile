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
# ENV PEXELS_KEY="aXA4IlmjYKdzM9R7JZX6l4SwVmxTsaJbMvp9l7jf7rE9VVbh5lbxvoKn"
# ENV OPENROUTER_API_KEY="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"
# ENV OPENAI_KEY="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"
ENV PEXELS_KEY="aXA4IlmjYKdzM9R7JZX6l4SwVmxTsaJbMvp9l7jf7rE9VVbh5lbxvoKn"
ENV OLLAMA_MODEL="gemma3:12b"
ENV OLLAMA_HOST="http://host.docker.internal:11434"

ENV FLASK_ENV=production
ENV TEMP=/tmp

# Expose the port the app runs on
EXPOSE 5050

# Run the application
CMD ["python", "app.py"]