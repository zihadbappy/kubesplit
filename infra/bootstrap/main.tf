# Long-lived pieces, applied once by a human with their own gcloud credentials:
# APIs, the image registry, the state bucket for infra/cluster, and keyless
# GitHub Actions access (Workload Identity Federation, no JSON keys).

data "google_project" "this" {}

locals {
  apis = [
    "artifactregistry.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "compute.googleapis.com",
    "container.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "sts.googleapis.com",
    "storage.googleapis.com",
  ]
}

resource "google_project_service" "apis" {
  for_each = toset(local.apis)
  service  = each.value
  # Turning an API off on destroy can break things outside this stack.
  disable_on_destroy = false
}

# -- Images -----------------------------------------------------------------

resource "google_artifact_registry_repository" "images" {
  repository_id = "kubesplit"
  location      = var.region
  format        = "DOCKER"
  description   = "KubeSplit images, pushed by GitHub Actions"

  # Every push adds a ~1.9 GB image; keep only the recent ones.
  cleanup_policy_dry_run = false
  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions {
      keep_count = var.keep_image_versions
    }
  }
  cleanup_policies {
    id     = "delete-the-rest"
    action = "DELETE"
    condition {
      tag_state  = "ANY"
      older_than = "86400s" # a day, so a fresh push is never deleted before KEEP applies
    }
  }

  depends_on = [google_project_service.apis]
}

# -- Terraform state for infra/cluster --------------------------------------

resource "google_storage_bucket" "tfstate" {
  name                        = "${var.project_id}-tfstate"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true # recover from a bad apply
  }
  lifecycle_rule {
    condition {
      num_newer_versions = 10
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.apis]
}

# -- CI identity: GitHub Actions -> service account, no keys ----------------

resource "google_service_account" "ci" {
  account_id   = "kubesplit-ci"
  display_name = "KubeSplit GitHub Actions"
  description  = "Pushes images and deploys the Helm chart. Reached only through WIF."

  depends_on = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github"
  display_name              = "GitHub Actions"

  depends_on = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github"
  display_name                       = "GitHub OIDC"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }
  # Only tokens minted for this repository are accepted at all.
  attribute_condition = "assertion.repository == \"${var.github_repository}\""
}

# Workflows from the repo may impersonate the CI service account.
resource "google_service_account_iam_member" "ci_wif" {
  service_account_id = google_service_account.ci.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repository}"
}

# Push images to this one repository, nothing else in Artifact Registry.
resource "google_artifact_registry_repository_iam_member" "ci_push" {
  repository = google_artifact_registry_repository.images.name
  location   = google_artifact_registry_repository.images.location
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.ci.email}"
}

# Deploy workloads with Helm: objects inside clusters, not the clusters
# themselves (creating or deleting clusters is a human's terraform apply).
resource "google_project_iam_member" "ci_deploy" {
  project = var.project_id
  role    = "roles/container.developer"
  member  = "serviceAccount:${google_service_account.ci.email}"
}
