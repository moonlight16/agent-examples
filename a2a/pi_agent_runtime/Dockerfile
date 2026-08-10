FROM node:24-alpine

ARG PI_REPOSITORY=https://github.com/kagenti/pi.git
ARG PI_COMMIT=95431d13aa9e5cd4ef3088a5bb2d060b03d7a072

RUN apk add --no-cache \
      build-base cairo-dev git giflib-dev jpeg-dev pango-dev pixman-dev \
      pkgconfig python3 py3-pip py3-virtualenv librsvg-dev && \
    python3 -m venv /opt/venv

RUN git clone "${PI_REPOSITORY}" /opt/pi && \
    cd /opt/pi && \
    git checkout "${PI_COMMIT}" && \
    npm ci && \
    npx tsgo -p packages/ai/tsconfig.build.json && \
    npm run build -w packages/agent && \
    npm run build -w packages/tui && \
    npm run build -w packages/coding-agent && \
    ln -s /opt/pi/packages/coding-agent/dist/cli.js /usr/local/bin/pi

WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
RUN /opt/venv/bin/pip install --no-cache-dir .

RUN mkdir -p /shared /home/node/.pi/agent && chown -R node:node /shared /home/node /app

ENV PATH=/opt/venv/bin:$PATH \
    HOME=/home/node \
    HOST=0.0.0.0 \
    PORT=8000 \
    PI_WORKSPACE=/shared \
    PI_AGENT_DIR=/home/node/.pi/agent

USER node
EXPOSE 8000
CMD ["pi-agent-runtime"]
