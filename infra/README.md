# Infrastructure (GCP)

Two Terraform stacks with different lifetimes:

| Stack | Lifetime | Creates | State |
|---|---|---|---|
| `bootstrap/` | once per project | APIs, Artifact Registry, state bucket, GitHub OIDC trust, CI service account | local (it creates the bucket) |
| `cluster/` | per benchmark session | zonal GKE cluster, Spot node pool, node service account | GCS bucket from `bootstrap/` |

## bootstrap: keyless CI

GitHub Actions never holds a GCP key. Each workflow run gets a short-lived
OIDC token from GitHub; Google's Security Token Service trades it for GCP
credentials through a Workload Identity pool, but only if the token says it
came from this repository (`attribute_condition`). The workflow then acts as
the `kubesplit-ci` service account, which can push to the one image repository
and deploy into clusters (`roles/container.developer`), nothing more.

```bash
cd infra/bootstrap
cp terraform.tfvars.example terraform.tfvars   # set project_id, github_repository
gcloud auth application-default login          # your own credentials, once
terraform init && terraform apply
terraform output -raw github_variables | bash  # sets the repo variables CI reads
```

Artifact Registry keeps the 10 most recent images and deletes older ones, since
each image is ~1.9 GB (CPU torch plus CIFAR-10).

## cluster: create, benchmark, destroy

A zonal GKE Standard cluster (one control plane, covered by GKE's free tier)
with one node pool of 3 `n2-standard-4` Spot VMs: room for the server (2 CPU),
4 clients (1 CPU each) and kube-prometheus-stack. N2 rather than E2 for steadier
CPU, since the benchmarks time CPU-bound work. Spot VMs can be reclaimed at any
time; if a node disappears mid-benchmark, rerun that configuration.

Nodes run as `kubesplit-nodes`, a service account that can write logs and
metrics and pull from the one image repository, instead of the project's
default compute account. Deletion protection is off: the cluster is meant to be
destroyed after each session.

```bash
cd infra/cluster
cp terraform.tfvars.example terraform.tfvars   # set project_id
terraform init -backend-config="bucket=<project_id>-tfstate"
terraform apply                                  # ~10 min
terraform output -raw github_variables | bash   # tells the deploy workflow where to go
$(terraform output -raw get_credentials)         # kubectl now talks to GKE
# ...benchmarks...
terraform destroy
```

## Checks

`make tf-check` runs what CI runs: `terraform fmt -check` and `validate` for both
stacks, without credentials or a backend.
