# Serves the API and the web form on one port. Hugging Face Spaces (Docker SDK) expects port 7860
# and a non-root user with uid 1000.
FROM python:3.11-slim

RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user src ./src
COPY --chown=user app ./app
COPY --chown=user models ./models
COPY --chown=user web ./web

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
