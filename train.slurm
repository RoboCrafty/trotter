#!/bin/bash
# Train Trotter as a batch job: keeps running when you disconnect, logs to a file.
#
#   sbatch train.slurm                                # fresh run
#   sbatch train.slurm --agent.resume True \
#       --agent.load-run <run folder> --agent.load-checkpoint model_<N>.pt
#
#   sbatch --export=ALL,TROTTER_VARIANT=max04 train.slurm   # a sweep variant
#
# Extra arguments are passed straight to `train`. Variants are defined at the
# end of src/trotter_rl/tasks/velocity_env_cfg.py; the run folder gets the
# variant's name.
# Watch it:  tail -f outputs/slurm-<JOBID>.out      Stop it:  scancel <JOBID>
#SBATCH --job-name=trotter_rl
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=05:00:00                  # 10k iterations take ~3 h at ~1.05 s each
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err

unset ${!SSL_*}

cd /sharchome/ft-rpd-ced-inw-de/fuzail/trotter
mkdir -p outputs

# Python block-buffers stdout when it's a file; flush each line so the log updates live.
export PYTHONUNBUFFERED=1
# Warp's kernel cache breaks on the shared /sharchome filesystem; keep it on local disk.
export WARP_CACHE_PATH=/tmp/$USER/warp_cache

uv run train Trotter-Velocity-Flat --env.scene.num-envs 8192 \
    ${TROTTER_VARIANT:+--agent.run-name "$TROTTER_VARIANT"} "$@"
