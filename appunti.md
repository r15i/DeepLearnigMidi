# dataset location is in 
./datasets

# dataset Lack (clean_midi)
it is a directory full of directories in which there are the midi files
**REMEMBER**: for the class RecursiveMidiDataset use the file *valid_paths* after scanning the *clean_midi* directory

# install using uv 
uv venv
source .venv/bin/activate.fish
uv pip install pandas numpy torch torchvision torchaudio --torch-backend=auto
