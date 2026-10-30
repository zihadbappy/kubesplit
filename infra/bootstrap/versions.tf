terraform {
  required_version = ">= 1.9"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 8.9"
    }
  }

  # State lives in the bucket this stack creates, next to the cluster's:
  #   terraform init -backend-config="bucket=<project_id>-tfstate"
  # In a new project the bucket doesn't exist yet: comment this block out,
  # apply once with local state, restore it, then move the state in with
  #   terraform init -migrate-state -backend-config="bucket=<project_id>-tfstate"
  backend "gcs" {
    prefix = "bootstrap"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
