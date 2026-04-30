# Remote compute on the LAN Win11 box

Treats `192.168.1.213` as a compute backend reached over SSH. Captured frames
stay off GitHub — they live in `captured_frames/` (gitignored) on both boxes
and sync via rsync.

The Mac side is the orchestrator. The Win11 side is "just a machine with the
repo on it"; it doesn't need a long-running daemon.

## One-time setup

### On Win11
1. Enable OpenSSH server:
   ```powershell
   # PowerShell as admin
   Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
   Start-Service sshd
   Set-Service -Name sshd -StartupType 'Automatic'
   New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server (sshd)' \
     -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
   ```
2. Make sure the repo is checked out (the Win11 Claude probably already did
   this) and `.venv` exists with the same dependencies as the Mac side. The
   easiest way: replicate `pip install -r tools/requirements.txt` plus
   whichever extras the active pipeline needs (DA3, gsplat, etc.).

### On Mac
1. Generate / reuse an SSH key, copy the public key to Win11:
   ```bash
   ssh-copy-id -i ~/.ssh/id_ed25519.pub user@192.168.1.213
   # if ssh-copy-id is missing:
   cat ~/.ssh/id_ed25519.pub | ssh user@192.168.1.213 \
     'mkdir -p .ssh && cat >> .ssh/authorized_keys'
   ```
2. Copy the config template and fill in your username and repo path:
   ```bash
   cp tools/remote/config.sh.example tools/remote/config.sh
   $EDITOR tools/remote/config.sh
   ```
3. Sanity check:
   ```bash
   tools/remote/remote.sh check
   ```

## Daily use

```bash
# Push the captured frames once (or after each new capture session).
tools/remote/remote.sh sync-frames

# Run a heavy job there. Whatever you'd type locally goes after `run` or `py`.
tools/remote/remote.sh py tools/cache_model_raw.py 20260427_123809 \
  --model-version v3
tools/remote/remote.sh py tools/gsplat_depth.py 20260427_123809 \
  --densify-stride 4 --densify-keyframes 64 --photo-iters 200

# Pull the standard derived dirs for one session back to the Mac.
tools/remote/remote.sh pull-artifacts 20260427_123809

# Then voxelize / view locally as usual.
```

`run` runs an arbitrary command, `py` is a thin wrapper that picks the venv's
python (`.venv/Scripts/python.exe` on git-bash, `.venv/bin/python` on WSL).
`shell` drops into an interactive session if you need to debug something
hands-on.

## Notes

- `config.sh` is gitignored; the example template is committed.
- The router has `192.168.1.213` pinned to the Win11 box, so the IP shouldn't
  drift. If it does, edit `WIN11_HOST` in `config.sh`.
- Two Claude Code instances can't talk to each other directly. The Mac one
  drives the pipeline via SSH; the Win11 one is for hands-on work on that box.
- If you'd rather use WSL on the Win11 side (CUDA does work in WSL2), set
  `WIN11_SHELL=wsl` and adjust `WIN11_REPO` to the Linux-side path.
