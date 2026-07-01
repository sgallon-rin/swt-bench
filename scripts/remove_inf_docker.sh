#!/usr/bin/env bash
# Remove all existing Docker images for swt-inference
# Use this script if you want to rebuild them

set -euo pipefail

delete_matching_images() {
  local label="$1"
  local pattern="$2"

  echo "============================================================"
  echo "Searching for Docker images: ${label}"
  echo "Pattern: ${pattern}"
  echo

  local images
  images=$(
    docker images --format '{{.Repository}}:{{.Tag}}' |
      grep -E "${pattern}" || true
  )

  if [[ -z "${images}" ]]; then
    echo "No matching Docker images found for: ${label}"
    echo
    return 0
  fi

  local count
  count=$(echo "${images}" | wc -l | tr -d ' ')

  echo "Found ${count} matching Docker image(s) for: ${label}"
  echo
  echo "The following Docker images will be removed:"
  echo
  echo "${images}" | sed 's/^/  - /'
  echo

  local confirm
  read -r -p "Delete these ${count} image(s) for ${label}? [y/N] " confirm

  case "${confirm}" in
    y|Y|yes|YES)
      echo
      echo "Removing ${count} image(s) for: ${label}"
      echo "${images}" | xargs -r docker rmi
      echo "Done."
      ;;
    *)
      echo
      echo "Skipped: ${label}"
      ;;
  esac

  echo
}

delete_matching_images "swt-inf.eval.*" '^swt-inf\.eval\.'
delete_matching_images "swt-inf-skill.eval.*" '^swt-inf-skill\.eval\.'

echo "Finished."