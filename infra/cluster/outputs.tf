output "cluster_name" {
  value = google_container_cluster.this.name
}

output "location" {
  value = google_container_cluster.this.location
}

output "get_credentials" {
  description = "Point kubectl at the cluster."
  value       = "gcloud container clusters get-credentials ${google_container_cluster.this.name} --zone ${google_container_cluster.this.location} --project ${var.project_id}"
}

output "github_variables" {
  description = "Run once so the deploy workflow knows which cluster to use."
  value       = <<-EOT
    gh variable set GKE_CLUSTER --body "${google_container_cluster.this.name}"
    gh variable set GKE_LOCATION --body "${google_container_cluster.this.location}"
  EOT
}
