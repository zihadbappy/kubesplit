terraform {
  required_version = ">= 1.9"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 8.9"
    }
  }

  # State lives in the bucket infra/bootstrap created:
  #   terraform init -backend-config="bucket=<project_id>-tfstate"
  backend "gcs" {
    prefix = "cluster"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
