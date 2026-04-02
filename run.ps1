docker run --rm -it `
    --gpus all `
    -e LD_LIBRARY_PATH=/usr/lib/wsl/lib `
    -v "$(pwd)/checkpoints:/workspace/mvdust3r/checkpoints" `
    -v "$env:USERPROFILE/.cache/torch:/root/.cache/torch" `
    -p 7860:7860 `
    mvdust3r:cu124 `
      --weights ./checkpoints/MVDp_s2.pth `
      --server_name 0.0.0.0 `
      --server_port 7860