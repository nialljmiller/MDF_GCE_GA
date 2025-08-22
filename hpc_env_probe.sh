#!/usr/bin/env bash
set -euo pipefail

OUT="hpc_diag_$(hostname)_$(date +%Y%m%d_%H%M%S).txt"
exec >"$OUT" 2>&1

# Helper: run only if command exists
run() { command -v "$1" >/dev/null 2>&1 && { echo "\n# $*"; "$@"; } || true; }

echo "===== BASIC ====="
echo "# date"; date
echo "# whoami / id"; whoami; id
echo "# hostname"; hostname
run hostnamectl || { echo "# uname -a"; uname -a; }
[ -r /etc/os-release ] && { echo "# /etc/os-release"; cat /etc/os-release; }

echo -e "\n===== SHELL ====="
echo "# SHELL, bash version, $0"; printf "%s\n" "$SHELL"; bash --version | head -n1; printf "%s\n" "$0"
echo "# shopt -op"; shopt -op || true
echo "# set -o (subset)"; set -o | grep -E '^(braceexpand|emacs|errexit|histexpand|history|monitor|nounset|pipefail|vi)' || true

echo -e "\n# Term/locale"
echo "TERM=$TERM"; run infocmp; run tput cols; run tput lines
run locale

echo -e "\n# PATH (split)"
echo "$PATH" | tr ':' '\n' | nl -ba

echo -e "\n# Key env vars (sanitized)"
env | grep -E '^(PATH|LD_LIBRARY_PATH|MODULEHOME|MODULEPATH|CONDA_|VIRTUAL_ENV|PYTHONPATH|SLURM_)' | sort

echo -e "\n===== DOTFILES (metadata + safe snippets) ====="
for f in ~/.bashrc ~/.bash_profile ~/.profile ~/.bash_login ~/.inputrc; do
  [ -f "$f" ] || continue
  echo -e "\n# File: $f"
  stat -c 'size=%s bytes mtime=%y' "$f" 2>/dev/null || ls -l "$f"
  echo "# First 200 non-comment lines (aliases, PS1, modules; secrets filtered)"
  # Show typical shell customizations but try to dodge obvious secrets
  sed -e 's/\t/    /g' "$f" \
  | grep -n -E '^(alias|PS1=|PROMPT_COMMAND=|module |source |\. |shopt |bind |complete )' \
  | grep -Ev '(API|TOKEN|SECRET|KEY|PASS|PASSWORD|AUTH|BEARER)' \
  | head -n 200 || true
done

echo -e "\n===== MODULES ====="
run module --version
echo "# module list"; module list 2>&1 || true
echo "# module avail (first 200 lines)"; module avail 2>&1 | head -n 200 || true
echo "# MODULEPATH"; echo "$MODULEPATH" | tr ':' '\n' || true

echo -e "\n===== COMPILERS / TOOLCHAIN ====="
run gcc --version | head -n1
run clang --version | head -n1
run icc --version | head -n1
run icx --version | head -n1
run cc --version | head -n1
run make --version | head -n1
run cmake --version
run ninja --version

echo -e "\n===== MPI / SLURM ====="
run mpicc -v
run mpirun --version | head -n2
run srun --version
run salloc --version
run sbatch --version
echo "# sinfo (compact)"; sinfo -o '%P %D %N %G %m %c %t' 2>&1 || true
echo "# squeue (your jobs)"; squeue -u "$USER" -o '%i %t %j %P %M %l %D %R' 2>&1 || true
echo "# scontrol show config (key lines)"; scontrol show config 2>/dev/null | egrep -i 'SlurmctldHost|SelectType|SchedulerType|Accounting|DefMemPer|MaxJobCount|Partitions' || true

echo -e "\n===== PYTHON / PKG MGMT ====="
run python --version
run python3 --version
run pip --version
run pip3 --version
run conda info
run mamba --version
run virtualenv --version
run poetry --version
run pipx --version

echo -e "\n===== GPU / ACCEL ====="
run nvidia-smi
run nvidia-smi -L
run nvcc --version
run rocminfo
run rocm-smi

echo -e "\n===== COMMON CLI TOOLS ====="
for t in tmux screen fzf rg fd bat eza exa tree ag delta git gh less wget curl rsync unzip zip tar jq htop bpytop top du df lsof file which whereis time gawk mawk sed awk perl node npm go rustc cargo; do
  if command -v "$t" >/dev/null 2>&1; then
    v=$("$t" --version 2>/dev/null | head -n1); [ -z "$v" ] && v=$("$t" -V 2>/dev/null | head -n1)
    echo "$t: ${v:-installed}"
  fi
done

echo -e "\n===== RESOURCES / LIMITS ====="
run ulimit -a
run nproc
run lscpu
run lshw -short
run free -h

echo -e "\n===== FILESYSTEM / QUOTAS ====="
echo "# df -h (home, pwd, /tmp)"
df -h "$HOME" "$PWD" /tmp 2>/dev/null || df -h
run quota -s
run lfs df -h
run lfs quota -hu "$USER"

echo -e "\n===== EDITORS ====="
run vim --version | head -n2
run nvim --version | head -n2
run emacs --version | head -n1
run code --version

echo -e "\n===== GIT ====="
run git --version
git config --global -l || true
git config -l || true

echo -e "\n===== TERM APPS (prompt frameworks if present) ====="
run starship --version
run powerline-daemon --version

echo -e "\n===== DONE ====="
echo "Report written to: $OUT"
