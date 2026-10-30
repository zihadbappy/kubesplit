variable "project_id" {
  description = "GCP project to create everything in."
  type        = string
}

variable "region" {
  description = "Region for Artifact Registry and the state bucket. Keep it the cluster's region."
  type        = string
  default     = "us-central1"
}

variable "github_repository" {
  description = "owner/name of the GitHub repo whose Actions may act as the CI service account."
  type        = string

  validation {
    condition     = can(regex("^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", var.github_repository))
    error_message = "Use the form owner/name, e.g. octocat/kubesplit."
  }
}

variable "keep_image_versions" {
  description = "Artifact Registry keeps this many most recent image versions; older ones are deleted."
  type        = number
  default     = 10
}
