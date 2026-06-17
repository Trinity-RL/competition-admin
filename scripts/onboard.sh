#!/usr/bin/env bash
# Grants a competitor read access to the demo repo and creates/grants
# them push access to their own private submission repo.
set -euo pipefail

USERNAME="${1:?usage: onboard.sh <github-username>}"
ORG="${ORG:?ORG env var required}"
DEMO_REPO="${DEMO_REPO:?DEMO_REPO env var required}"
TEMPLATE_REPO="${TEMPLATE_REPO:?TEMPLATE_REPO env var required (owner/name)}"

# GitHub usernames: alphanumeric and hyphens only, max 39 chars.
# Reject anything else before it touches the GH API.
if [[ ! "$USERNAME" =~ ^[A-Za-z0-9-]{1,39}$ ]]; then
  echo "Refusing to process invalid GitHub username: $USERNAME" >&2
  exit 1
fi

SUBMISSION_REPO="submission-${USERNAME}"

echo "Onboarding ${USERNAME}..."

# 1. Read-only access to the shared demo repo.
gh api -X PUT "repos/${ORG}/${DEMO_REPO}/collaborators/${USERNAME}" -f permission=pull

# 2. Create their private submission repo from the template, if needed.
if gh api "repos/${ORG}/${SUBMISSION_REPO}" >/dev/null 2>&1; then
  echo "Repo ${SUBMISSION_REPO} already exists, skipping creation."
else
  gh api -X POST "repos/${TEMPLATE_REPO}/generate" \
    -f owner="${ORG}" \
    -f name="${SUBMISSION_REPO}" \
    -F private=true \
    -F include_all_branches=false
fi

# 3. Push access to their own submission repo only.
gh api -X PUT "repos/${ORG}/${SUBMISSION_REPO}/collaborators/${USERNAME}" -f permission=push

# 4. Grant the submission repo access to the org-level COMPETITION_DISPATCH_TOKEN secret
#    so their compete.yml can trigger evaluation on competition-admin.
REPO_ID=$(gh api "repos/${ORG}/${SUBMISSION_REPO}" --jq '.id')
EXISTING_IDS=$(gh api "orgs/${ORG}/actions/secrets/COMPETITION_DISPATCH_TOKEN/repositories" \
  --jq '[.repositories[].id]' 2>/dev/null || echo "[]")
UPDATED_IDS=$(echo "$EXISTING_IDS" | python3 -c "
import json, sys
ids = json.load(sys.stdin)
new_id = int('${REPO_ID}')
if new_id not in ids:
    ids.append(new_id)
print(json.dumps(ids))
")
gh api -X PUT "orgs/${ORG}/actions/secrets/COMPETITION_DISPATCH_TOKEN/repositories" \
  --input <(echo "{\"selected_repository_ids\": ${UPDATED_IDS}}")

echo "submission_repo_url=https://github.com/${ORG}/${SUBMISSION_REPO}"
