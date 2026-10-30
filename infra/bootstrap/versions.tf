terraform {
  required_version = ">= 1.9"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 8.9"
    }
  }

  # Local state on purpose: this stack creates the bucket that holds the
  # cluster stack's state. Keep terraform.tfstate safe (it is gitignored).
}

provider "google" {
  project = var.project_id
  region  = var.region
}
