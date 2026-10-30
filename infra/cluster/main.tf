# The benchmark cluster: created for a session, destroyed after.
#   terraform apply   ...run benchmarks...   terraform destroy

# Nodes run as their own service account with only what a node needs, rather
# than the project's default compute account (often Editor on the project).
resource "google_service_account" "nodes" {
  account_id   = "${var.cluster_name}-nodes"
  display_name = "KubeSplit GKE nodes"
}

locals {
  node_roles = [
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer",
  ]
}

resource "google_project_iam_member" "nodes" {
  for_each = toset(local.node_roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.nodes.email}"
}

# Pull images from the bootstrap's repository, and only that one.
resource "google_artifact_registry_repository_iam_member" "nodes_pull" {
  repository = "kubesplit"
  location   = var.region
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.nodes.email}"
}

resource "google_container_cluster" "this" {
  name     = var.cluster_name
  location = var.zone # a zone, not a region: one control plane, one zone of nodes

  # Node pools are managed separately below; GKE insists on creating a
  # default pool first, so make it tiny and remove it right away. Provider 8.x
  # marks this deprecated: from GKE 1.36.4-gke.1469000, initial_node_count = 0
  # (and no node_config) does the same. Kept because it works on any version
  # the release channel may pick; validate prints a warning about it.
  remove_default_node_pool = true
  initial_node_count       = 1
  node_config {
    service_account = google_service_account.nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  deletion_protection = false # ephemeral by design
  networking_mode     = "VPC_NATIVE"
  ip_allocation_policy {}

  release_channel {
    channel = "REGULAR"
  }
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # kube-prometheus-stack does the metrics; keep Google's agents to the minimum.
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS"]
  }
  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS"]
    managed_prometheus {
      enabled = false
    }
  }

  depends_on = [google_project_iam_member.nodes]
}

resource "google_container_node_pool" "workers" {
  name       = "workers"
  cluster    = google_container_cluster.this.id
  location   = var.zone
  node_count = var.node_count

  node_config {
    machine_type    = var.machine_type
    spot            = var.spot
    disk_type       = "pd-balanced"
    disk_size_gb    = 50 # the image is ~1.9 GB; leave room for a few versions
    service_account = google_service_account.nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    labels = {
      pool = "workers"
    }
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    shielded_instance_config {
      enable_secure_boot = true
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}
