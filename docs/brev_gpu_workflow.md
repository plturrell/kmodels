# Brev GPU Workflow

This repository includes a helper script to spin up a Brev GPU instance, sync the codebase, and drop into a shell whenever you need to run accelerated training.

## Prerequisites

- Install the Brev CLI: `brew install brevdev/homebrew-brev/brev`
- Authenticate: `brev login` (confirm `brev whoami` shows the correct account)
- Optionally pick an org: `brev org ls` + `brev set <org-name>`

All commands below assume you run them from the repo root (`/Users/user/Documents/competitions`).

## One-time instance creation

```bash
./scripts/brev_gpu.sh create
```

Defaults:

- Instance name: `competitions-gpu`
- GPU shape: `n1-highmem-4:nvidia-tesla-t4:1`
- Remote sync path: `~/projects/competitions`

Override the GPU (or CPU) shape by exporting environment variables before running the script:

```bash
export BREV_GPU_TYPE=a2-highgpu-1g:nvidia-a100-40gb:1
export BREV_CPU_TYPE=8x32
./scripts/brev_gpu.sh create my-fast-gpu
```

## Per-project entry points

Each competition folder includes a wrapper script (for example `cafa_6_protein_function_prediction/brev_gpu.sh`). The wrapper sets `BREV_LOCAL_DIR`/`BREV_REMOTE_DIR` to project-scoped paths before delegating to the repo helper so you can sync just that project and keep remote directories organised.

- Run the wrapper from inside the project to operate on its files only.
- Export project-specific overrides (e.g. `BREV_INSTANCE_NAME`) right before invoking the wrapper if you want dedicated instances per competition.

Example session:

```bash
cd competitions/csiro_biomass
./brev_gpu.sh create      # one-time provisioning
./brev_gpu.sh sync-up     # copy the project to Brev
./brev_gpu.sh shell       # open the GPU workspace
```

Use the top-level `./scripts/brev_gpu.sh` when you need to manage the entire monorepo or supply custom paths manually.

## Typical on-demand workflow

1. **Start the workspace (if it was stopped)**  
   ```bash
   ./scripts/brev_gpu.sh start
   ```

2. **Sync the latest local code to the instance**  
   ```bash
   ./scripts/brev_gpu.sh sync-up
   ```
   - Add a subdirectory (relative to the repo) to sync only part of the project: `./scripts/brev_gpu.sh sync-up "" mabe_mouse_behavior_detection`.
   - Override the remote path with a final argument: `./scripts/brev_gpu.sh sync-up "" "" ~/experiments/competitions`.

3. **Open an interactive shell on the GPU box**  
   ```bash
   ./scripts/brev_gpu.sh shell
   ```
   Install dependencies (conda/pip) and launch your training jobs as needed.

4. **Pull results back to your Mac (optional)**  
   ```bash
   ./scripts/brev_gpu.sh sync-down "" outputs
   ```
   This copies `~/projects/competitions` (or your overridden remote path) back into `./outputs` locally, creating the folder if needed.

5. **Stop the instance when you're done**  
   ```bash
   ./scripts/brev_gpu.sh stop
   ```

Remove an instance completely with `./scripts/brev_gpu.sh delete`.

## Helpful environment overrides

- `BREV_INSTANCE_NAME`: default instance name
- `BREV_GPU_TYPE`: GPU shape to request on creation
- `BREV_CPU_TYPE`: CPU shape to pair with the GPU
- `BREV_LOCAL_DIR`: local directory (absolute or repo-relative) to sync
- `BREV_REMOTE_DIR`: remote destination directory

Example: keep a staging branch synced to a different workspace.

```bash
export BREV_INSTANCE_NAME=competitions-staging
export BREV_GPU_TYPE=a2-megagpu-1g:nvidia-a100-80gb:1
export BREV_REMOTE_DIR=~/projects/competitions-staging
./scripts/brev_gpu.sh create
./scripts/brev_gpu.sh sync-up
```

## Troubleshooting

- Run `brev status <instance>` if a command complains about the instance state.
- If `brev copy` fails because the remote directory was deleted, create it inside the Brev shell (`mkdir -p ~/projects/competitions`) and retry `sync-up`.
- Use `brev ls` anytime to see all workspaces tied to the active org.
