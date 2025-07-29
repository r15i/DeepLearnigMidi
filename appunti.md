# install using uv 
uv venv
source .venv/bin/activate.fish
uv pip install torch torchvision torchaudio --torch-backend=auto
