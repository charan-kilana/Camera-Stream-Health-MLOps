# Camera Stream Health MLOps

A basic end-to-end machine learning project that predicts whether a surveillance
camera stream is at risk of failure from operational telemetry.

The repository follows a simple MLOps workflow:

1. Generate a reproducible synthetic dataset.
2. Train and evaluate a Random Forest classifier.
3. Save the trained model.
4. Serve custom predictions through FastAPI and Docker.
5. Build and publish the API image to Amazon ECR.
6. Deploy the image as a standard Kubernetes Deployment and Service.
7. Alternatively, serve the S3 model through KServe on a local KIND cluster.

## Model inputs

- Frames per second (`fps`)
- Network latency in milliseconds
- Packet loss percentage
- Video bitrate in Kbps
- Recent reconnection count
- Current stream uptime in hours

The target, `stream_failure`, indicates whether a stream fails within the chosen
prediction window. The dataset is a domain-informed synthetic approximation for
demonstrating the ML workflow, not production camera telemetry.

## Run locally

```bash
python -m venv .venv
pip install -r requirements.txt
python generate_data.py
python train.py
python api.py
```

Open `http://localhost:8000/docs` to test the API.

## Run with Docker

Generate the dataset and train the model before building the image:

```bash
python generate_data.py
python train.py
docker build -t camera-stream-health-api:latest .
docker run --rm -p 8000:8000 --name camera-health-api camera-stream-health-api:latest
```

Open `http://localhost:8000/docs` for Swagger UI or check container health:

```bash
curl http://localhost:8000/health
docker ps
```

The Docker image includes `models/stream_failure_model.pkl`. The model is kept
out of Git and must be trained or downloaded before each image build.

## Continuous integration

On the `k8s-deploy` branch, GitHub Actions recreates the dataset, trains the
model, builds the FastAPI image, and pushes two tags to Amazon ECR:

- The Git commit SHA, used as the immutable Kubernetes deployment tag.
- `latest`, provided as a convenient development tag.

The workflow then updates `k8s/deployment.yaml` with the immutable image tag and
commits it back using `[skip ci]`. The workflow requires the repository secrets
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. The AWS identity must be able to
authenticate to ECR and push to the `camera-stream-health-api` repository.

## Kubernetes image deployment on KIND

This path deploys `api.py` and the trained model packaged together inside the
Docker image. KServe, cert-manager, S3 model downloading, and
`k8s/inference.yaml` are not involved.

### 1. Build and publish the image

Push a commit to `k8s-deploy`, then wait for the **Build and Publish Kubernetes
Image** workflow to finish. Pull the workflow's manifest commit before applying
it locally:

```bash
git switch k8s-deploy
git pull origin k8s-deploy
```

The image is published under:

```text
830283279505.dkr.ecr.us-east-1.amazonaws.com/camera-stream-health-api
```

### 2. Create a separate KIND cluster

```bash
kind create cluster --name camera-api --image kindest/node:v1.32.2
kubectl config use-context kind-camera-api
kubectl get nodes
```

### 3. Create the namespace and ECR pull secret

The ECR repository is private, so the cluster needs a registry credential. This
command uses the AWS CLI credentials on your laptop and stores the short-lived
ECR token only as a Kubernetes Secret:

```bash
kubectl apply -f k8s/namespace.yaml

kubectl create secret docker-registry ecr-registry-secret \
  --namespace camera-health \
  --docker-server=830283279505.dkr.ecr.us-east-1.amazonaws.com \
  --docker-username=AWS \
  --docker-password="$(aws ecr get-login-password --region us-east-1)" \
  --dry-run=client -o yaml | kubectl apply -f -
```

ECR login tokens expire, so rerun the Secret command if image pulling later
starts failing with an authorization error.

### 4. Deploy the API

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

kubectl rollout status deployment/camera-stream-health-api \
  --namespace camera-health --timeout=300s

kubectl get pods,service -n camera-health
```

### 5. Access and test the API

Keep the port-forward running in one terminal:

```bash
kubectl port-forward -n camera-health \
  service/camera-stream-health-api 8080:80
```

From another terminal, test the health and prediction endpoints:

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

Unlike the KServe endpoint, this custom FastAPI endpoint returns
`stream_failure`, `failure_probability`, and `risk_level`.

### 6. Cleanup

```bash
kubectl delete namespace camera-health
kind delete cluster --name camera-api
```

## Inference options

The project demonstrates two independent serving paths:

- **FastAPI and Docker** load the local model through `api.py` and return a
  failure prediction, probability, and risk level.
- **KServe** downloads the model directly from Amazon S3 and serves it through
  KServe's built-in scikit-learn runtime. This path does not use `api.py` or the
  project Docker image and returns the generic `predictions` response.

## KServe deployment on KIND

The tested local environment uses Kubernetes 1.32.2, cert-manager, KServe
0.16.0, and KServe `RawDeployment` mode.

### 1. Create the KIND cluster

```bash
kind create cluster --name camera-health --image kindest/node:v1.32.2
kubectl config use-context kind-camera-health
kubectl get nodes
```

### 2. Install cert-manager

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/latest/download/cert-manager.yaml

kubectl wait --for=condition=Available deployment --all \
  --namespace cert-manager --timeout=300s
```

### 3. Install the KServe CRDs

```bash
kubectl create namespace kserve

helm install kserve-crd oci://ghcr.io/kserve/charts/kserve-crd \
  --version v0.16.0 \
  --namespace kserve \
  --wait
```

### 4. Install the KServe controller

```bash
helm install kserve oci://ghcr.io/kserve/charts/kserve \
  --version v0.16.0 \
  --namespace kserve \
  --set kserve.controller.deploymentMode=RawDeployment \
  --wait
```

Verify the controller:

```bash
kubectl get pods -n kserve
```

### 5. Configure private S3 access

Apply the namespace, empty Secret template, and ServiceAccount:

```bash
kubectl apply -f k8s/serviceaccount.yaml
```

Keep AWS credentials out of Git. Enter them only in the local shell and replace
the empty Kubernetes Secret:

```bash
read -p "AWS Access Key ID: " AWS_ACCESS_KEY_ID
read -s -p "AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
echo

kubectl create secret generic s3-secret \
  --namespace ml-eagle \
  --from-literal=AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  --from-literal=AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl annotate secret s3-secret --namespace ml-eagle \
  serving.kserve.io/s3-endpoint=s3.amazonaws.com \
  serving.kserve.io/s3-usehttps="1" \
  serving.kserve.io/s3-region=us-east-1 \
  --overwrite

unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
```

### 6. Deploy the camera stream model

The `InferenceService` downloads this model from S3:

```text
s3://charan-camera-stream-health-eaglesight/models/stream_failure_model.pkl
```

Deploy and check it:

```bash
kubectl apply -f k8s/inference.yaml
kubectl get inferenceservice camera-stream-health-predictor -n ml-eagle
kubectl get pods -n ml-eagle
kubectl get service -n ml-eagle
```

### 7. Port-forward the generated service

Keep this command running in one terminal:

```bash
kubectl port-forward -n ml-eagle \
  service/camera-stream-health-predictor-predictor 8080:80
```

### 8. Test KServe inference

Feature order is `fps`, `latency_ms`, `packet_loss_percent`, `bitrate_kbps`,
`reconnect_count`, and `uptime_hours`.

High-risk stream:

```bash
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"instances":[[8,650,12,700,5,3]]}' \
  http://localhost:8080/v1/models/camera-stream-health-predictor:predict
```

Expected response:

```json
{"predictions":[1]}
```

Healthy stream:

```bash
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"instances":[[29,40,0.2,4200,0,300]]}' \
  http://localhost:8080/v1/models/camera-stream-health-predictor:predict
```

Expected response:

```json
{"predictions":[0]}
```

### 9. Cleanup

```bash
kubectl delete inferenceservice camera-stream-health-predictor -n ml-eagle
kubectl delete namespace ml-eagle
helm uninstall kserve -n kserve
helm uninstall kserve-crd -n kserve
kubectl delete namespace kserve
kind delete cluster --name camera-health
```
