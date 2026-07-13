# Camera Stream Health MLOps

An end-to-end machine learning project that predicts whether a surveillance
camera stream is at risk of failure from operational telemetry.

The `k8s-deploy` branch demonstrates this workflow:

1. Generate a reproducible synthetic dataset.
2. Train and evaluate a Random Forest classifier.
3. Package the trained model and FastAPI application in a Docker image.
4. Push immutable and `latest` image tags to Docker Hub.
5. Deploy the image with a Kubernetes Deployment and Service.

The separate `kserve-deploy` branch demonstrates model serving through KServe.

## Model inputs

- Frames per second (`fps`)
- Network latency in milliseconds (`latency_ms`)
- Packet loss percentage (`packet_loss_percent`)
- Video bitrate in Kbps (`bitrate_kbps`)
- Recent reconnection count (`reconnect_count`)
- Current stream uptime in hours (`uptime_hours`)

The target, `stream_failure`, indicates whether a stream fails within the chosen
prediction window. The dataset is a domain-informed synthetic approximation for
demonstrating the ML workflow, not production camera telemetry.

## Run locally

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
python generate_data.py
python train.py
python api.py
```

Open `http://localhost:8000/docs` to use the Swagger UI.

## Build and test the image locally

The model is ignored by Git, so generate it before building locally:

```bash
python generate_data.py
python train.py
docker build -t camera-stream-health-api:local .
docker run --rm -p 8000:8000 --name camera-health-api \
  camera-stream-health-api:local
```

Test the container:

```bash
curl http://localhost:8000/health
```

## Docker Hub setup

Create a public Docker Hub repository named:

```text
camera-stream-health-api
```

Create a Docker Hub personal access token, then add these GitHub repository
secrets under **Settings > Secrets and variables > Actions**:

- `DOCKERHUB_USERNAME`: your Docker Hub username
- `DOCKERHUB_TOKEN`: your Docker Hub personal access token

Do not store a Docker Hub password or token in this repository.

## GitHub Actions pipeline

The workflow runs for pushes to `k8s-deploy` and performs the following steps:

1. Generates the dataset.
2. Trains and verifies `models/stream_failure_model.pkl`.
3. Logs in to Docker Hub.
4. Builds the Docker image, including the trained model.
5. Pushes `<username>/camera-stream-health-api:<git-sha>`.
6. Pushes `<username>/camera-stream-health-api:latest`.
7. Updates `k8s/deployment.yaml` with the immutable SHA tag.
8. Commits the updated manifest with `[skip ci]` to avoid a pipeline loop.

After the workflow succeeds, update your local branch with its manifest commit:

```bash
git pull origin k8s-deploy
```

## Deploy to a new KIND cluster

### 1. Create the cluster

```bash
kind create cluster --name camera-api --image kindest/node:v1.32.2
kubectl config use-context kind-camera-api
kubectl get nodes
```

### 2. Apply the manifests

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

kubectl rollout status deployment/camera-stream-health-api \
  --namespace camera-health --timeout=300s

kubectl get pods,service -n camera-health
```

Because the Docker Hub repository is public, KIND can pull the image without a
Docker login or Kubernetes registry Secret.

### 3. Port-forward the Service

Keep this command running in one terminal:

```bash
kubectl port-forward -n camera-health \
  service/camera-stream-health-api 8080:80
```

### 4. Test the deployed API

From another terminal:

```bash
curl http://localhost:8080/health

curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fps": 8,
    "latency_ms": 650,
    "packet_loss_percent": 12,
    "bitrate_kbps": 700,
    "reconnect_count": 5,
    "uptime_hours": 3
  }'
```

The response contains `stream_failure`, `failure_probability`, and `risk_level`.

### 5. Add Traefik Ingress

The direct Service port-forward above is sufficient for basic testing. To match
the Ingress flow from the reference project, install Traefik with its official
Helm chart:

```bash
helm repo add traefik https://traefik.github.io/charts
helm repo update

helm upgrade --install traefik traefik/traefik \
  --namespace traefik \
  --create-namespace \
  --wait

kubectl get pods,service -n traefik
```

Apply the application Ingress:

```bash
kubectl apply -f k8s/ingress.yaml
kubectl get ingress -n camera-health
```

On a cloud cluster, Traefik's `LoadBalancer` Service can receive an external
address. KIND has no cloud load balancer by default, so port-forward Traefik in
one terminal:

```bash
kubectl port-forward -n traefik service/traefik 8081:80
```

Test routing through the Ingress from another terminal. The `Host` header must
match `camera-health.local` from `k8s/ingress.yaml`:

```bash
curl -H "Host: camera-health.local" \
  http://localhost:8081/health

curl -X POST http://localhost:8081/predict \
  -H "Host: camera-health.local" \
  -H "Content-Type: application/json" \
  -d '{
    "fps": 8,
    "latency_ms": 650,
    "packet_loss_percent": 12,
    "bitrate_kbps": 700,
    "reconnect_count": 5,
    "uptime_hours": 3
  }'
```

### 6. Deploy a newer image

After a later pipeline run and `git pull`, apply the updated immutable tag:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl rollout status deployment/camera-stream-health-api \
  --namespace camera-health --timeout=300s
```

### 7. Cleanup

```bash
helm uninstall traefik -n traefik
kubectl delete namespace traefik
kubectl delete namespace camera-health
kind delete cluster --name camera-api
```

## Private Docker Hub repositories

Running `docker login` on the laptop does not automatically authenticate the
Docker containers that act as KIND nodes. For a private repository, create a
Kubernetes `docker-registry` Secret and reference it through `imagePullSecrets`
in the Deployment. This demo uses a public repository to keep local deployment
simple.
