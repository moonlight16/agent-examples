#!/bin/bash
set -euo pipefail

# Deploy kagenti_chat to Kagenti cluster
# Usage: ./deploy-to-kagenti.sh [namespace]
#
# Prerequisites:
#   - Kagenti cluster with Shipwright installed
#   - buildah-insecure-direct ClusterBuildStrategy installed
#   - Internal registry at registry.cr-system (ClusterIP 10.43.28.116:5000)
#   - k3s nodes configured to allow insecure registry access
#   - llm-d deployed with Llama-3.3-70B-Instruct model

NAMESPACE="${1:-team1}"
AGENT_NAME="kagenti-chat"
REPO_URL="https://github.com/moonlight16/agent-examples"
CONTEXT_DIR="a2a/kagenti_chat"
REGISTRY_IP="10.43.28.116:5000"
GATEWAY_HOST="kagenti-chat.163-75-85-180.sslip.io"

echo "==> Deploying ${AGENT_NAME} to namespace ${NAMESPACE}"

# Ensure namespace has label for shared gateway access
kubectl label namespace ${NAMESPACE} shared-gateway-access=true --overwrite

# 1. Create/update Build using buildah-insecure-direct strategy
echo "==> Creating Build..."
kubectl apply -f - <<EOF
apiVersion: shipwright.io/v1beta1
kind: Build
metadata:
  name: ${AGENT_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: ${AGENT_NAME}
    kagenti.io/framework: a2a
spec:
  source:
    type: Git
    git:
      url: ${REPO_URL}
      revision: main
    contextDir: ${CONTEXT_DIR}
  strategy:
    name: buildah-insecure-direct
    kind: ClusterBuildStrategy
  paramValues:
    - name: dockerfile
      value: Dockerfile
  output:
    image: registry.cr-system.svc.cluster.local:5000/${AGENT_NAME}:latest
  retention:
    failedLimit: 3
    succeededLimit: 3
  timeout: 15m
EOF

# 2. Trigger BuildRun
echo "==> Triggering BuildRun..."
BUILD_RUN=$(kubectl create -f - -o jsonpath='{.metadata.name}' <<EOF
apiVersion: shipwright.io/v1beta1
kind: BuildRun
metadata:
  generateName: ${AGENT_NAME}-run-
  namespace: ${NAMESPACE}
spec:
  build:
    name: ${AGENT_NAME}
EOF
)

echo "==> BuildRun created: ${BUILD_RUN}"
echo "==> Waiting for build to complete (this may take a few minutes)..."

kubectl -n ${NAMESPACE} wait --for=condition=Succeeded --timeout=10m buildrun/${BUILD_RUN} 2>&1 || {
  echo "ERROR: Build failed. Check logs:"
  echo "  kubectl -n ${NAMESPACE} logs -l buildrun.shipwright.io/name=${BUILD_RUN} -c step-build-and-push"
  exit 1
}

echo "==> Build succeeded!"

# 3. Deploy the agent (using ClusterIP for image to avoid DNS issues)
echo "==> Deploying agent workload..."
kubectl apply -f - <<EOF
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${AGENT_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: ${AGENT_NAME}
    kagenti.io/framework: a2a
    kagenti.io/inject: disabled
  annotations:
    kagenti.io/description: "General-purpose chat agent using llm-d and AG2"
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ${AGENT_NAME}
  template:
    metadata:
      labels:
        app: ${AGENT_NAME}
        kagenti.io/framework: a2a
        kagenti.io/inject: disabled
    spec:
      containers:
        - name: ${AGENT_NAME}
          image: ${REGISTRY_IP}/${AGENT_NAME}:latest
          imagePullPolicy: Always
          ports:
            - containerPort: 8000
              name: http
          env:
            - name: LOG_LEVEL
              value: "INFO"
            - name: A2A_HOST
              value: "0.0.0.0"
            - name: A2A_PORT
              value: "8000"
            - name: LLM_API_KEY
              value: "dummy"
            - name: LLM_API_BASE
              value: "http://ms-meta-llama-llama-3-3-70b-instruct-svc.llm-d.svc.cluster.local:8000/v1"
            - name: LLM_MODEL
              value: "meta-llama/Llama-3.3-70B-Instruct"
            - name: LLM_TEMPERATURE
              value: "0.2"
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "2000m"
              memory: "2Gi"
          livenessProbe:
            httpGet:
              path: /.well-known/agent.json
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /.well-known/agent.json
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: ${AGENT_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: ${AGENT_NAME}
spec:
  selector:
    app: ${AGENT_NAME}
  ports:
    - name: http
      port: 8000
      targetPort: 8000
  type: ClusterIP
---
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: ${AGENT_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: ${AGENT_NAME}
spec:
  parentRefs:
    - name: http
      namespace: kagenti-system
      kind: Gateway
  hostnames:
    - "${GATEWAY_HOST}"
  rules:
    - matches:
        - path:
            type: PathPrefix
            value: /
      backendRefs:
        - name: ${AGENT_NAME}
          port: 8000
EOF

echo "==> Waiting for deployment to be ready..."
kubectl -n ${NAMESPACE} rollout status deployment/${AGENT_NAME} --timeout=5m

echo ""
echo "==> Deployment complete!"
echo "    Agent URL: https://${GATEWAY_HOST}"
echo "    Agent card: https://${GATEWAY_HOST}/.well-known/agent-card.json"
echo ""
echo "Test with:"
echo "  curl -sk https://${GATEWAY_HOST}/.well-known/agent-card.json | jq ."
echo ""
echo "Send a chat message:"
echo "  curl -sk -X POST https://${GATEWAY_HOST}/ \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"jsonrpc\":\"2.0\",\"id\":\"1\",\"method\":\"message/send\",\"params\":{\"message\":{\"role\":\"user\",\"parts\":[{\"text\":\"Hello!\"}],\"messageId\":\"test-1\"}}}' | jq"
