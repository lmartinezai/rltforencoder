"""Copy a target store (or any directory) to the 3060 Ti machine over SFTP, skipping files already there with the same size, so an interrupted
copy simply continues.

    python experiments/small_study/push_store.py stores/fineweb_s42_50m <repo-on-second-machine>/stores/fineweb_s42_50m

Uses the same connection settings as lan_ssh.py (CODEX_SSH_HOST / CODEX_SSH_PASSWORD, host <lan-host> by default)."""
import os
import sys
from pathlib import Path

import paramiko

host = os.environ.get("CODEX_SSH_HOST", "<lan-host>")
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, username=os.environ.get("CODEX_SSH_USER", ""), password=os.environ.get("CODEX_SSH_PASSWORD", ""), timeout=30)
sftp = client.open_sftp()
local_root, remote_root = Path(sys.argv[1]), sys.argv[2].replace("\\", "/")


def mkdirs(path):
    parts, current = path.split("/"), ""
    for part in parts:
        current = f"{current}/{part}" if current else part
        if current.endswith(":"):
            continue
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)


sent = skipped = 0
for path in sorted(local_root.rglob("*")):
    if not path.is_file():
        continue
    remote = f"{remote_root}/{path.relative_to(local_root).as_posix()}"
    try:
        if sftp.stat(remote).st_size == path.stat().st_size:
            skipped += 1
            continue
    except OSError:
        pass
    mkdirs(remote.rsplit("/", 1)[0])
    sftp.put(str(path), remote + ".part")
    try:
        sftp.remove(remote)
    except OSError:
        pass
    sftp.rename(remote + ".part", remote)  # a file is complete only under its final name
    sent += 1
    if sent % 20 == 0:
        print(f"sent {sent} files, skipped {skipped}", flush=True)
print(f"done: sent {sent} files, skipped {skipped}", flush=True)
