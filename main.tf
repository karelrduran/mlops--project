terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 2.15"
    }
  }
}

provider "docker" {
  # registry_auth {
  #   address  = var.docker_registry
  #   username = var.docker_username
  #   password = var.docker_password
  # }
}

# Local variable to handle conditional tags
locals {
  tags = var.docker_registry != "" ? ["${var.docker_registry}/${var.mlflow-image}:latest"] : ["${var.mlflow-image}:latest"]
}

# Build Docker image
resource "docker_image" "app_image" {
  name = local.tags[0]

  build {
    context    = "./"  # Replace with path to your Dockerfile
    dockerfile = "Dockerfile"
  }
}

# Create Docker container
resource "docker_container" "app_service" {
  name  = "app-service"
  image = docker_image.app_image.image_id

  ports {
    internal = var.port
    external = var.port
  }
  # Optional: Additional service configurations like resources, environment variables, etc.
}



