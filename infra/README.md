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

## Checks

`make tf-check` runs what CI runs: `terraform fmt -check` and `validate` for both
stacks, without credentials or a backend.
