output "image_repository" {
  description = "Push and pull images here (append /kubesplit:<tag>)."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

output "state_bucket" {
  description = "Backend bucket for infra/cluster: terraform init -backend-config=bucket=<this>."
  value       = google_storage_bucket.tfstate.name
}

output "data_bucket" {
  description = "CI copies CIFAR-10 for the image from here; upload it once (infra/README.md)."
  value       = google_storage_bucket.data.name
}

output "workload_identity_provider" {
  description = "Full provider name for google-github-actions/auth."
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "ci_service_account" {
  value = google_service_account.ci.email
}

output "github_variables" {
  description = "Run these once so the workflows can reach GCP (repository variables, not secrets: none of this is sensitive)."
  value       = <<-EOT
    gh variable set GCP_PROJECT_ID --body "${var.project_id}"
    gh variable set GCP_REGION --body "${var.region}"
    gh variable set GCP_WIF_PROVIDER --body "${google_iam_workload_identity_pool_provider.github.name}"
    gh variable set GCP_CI_SERVICE_ACCOUNT --body "${google_service_account.ci.email}"
    gh variable set IMAGE_REPOSITORY --body "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}/kubesplit"
  EOT
}
