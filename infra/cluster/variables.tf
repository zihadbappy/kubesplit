variable "project_id" {
  description = "GCP project (the one infra/bootstrap set up)."
  type        = string
}

variable "region" {
  description = "Region of the bootstrap's Artifact Registry repository."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "A zonal cluster: one control plane, free under GKE's free tier."
  type        = string
  default     = "us-central1-a"
}

variable "cluster_name" {
  type    = string
  default = "kubesplit"
}

variable "machine_type" {
  description = "N2 rather than E2: steadier CPU, which benchmark timings depend on."
  type        = string
  default     = "n2-standard-4"
}

variable "node_count" {
  description = "Server (2 CPU) + 4 clients (1 CPU each) + monitoring fit on 3 x 4 vCPU."
  type        = number
  default     = 3
}

variable "spot" {
  description = "Spot VMs cost far less but can be reclaimed at any time; rerun a benchmark if a node goes."
  type        = bool
  default     = true
}
