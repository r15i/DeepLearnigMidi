import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import DBAdapters as dba

# 1. Dataset Selection
csv_file_path = "./dataset/MAESTRO_Dataset/maestro-v3.0.0.csv"
midi_base_path = "./dataset/MAESTRO_Dataset/maestro-v3.0.0"
my_dataset = dba.MaestroMIDIDataset(
    csv_file=csv_file_path, midi_base_path=midi_base_path
)

# TODO: implement the same for the other dataset


# 3. Create a DataLoader
# batch_size = 4
# shuffle = True  # Usually True for training, False for validation/testing
# num_workers = (
#     0  # Set to >0 for multi-process data loading (e.g., 2, 4, 8 depending on CPU cores)
# )
#
# data_loader = DataLoader(
#     my_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
# )

# 4. Iterate through the DataLoader (e.g., in a training loop)
# print("\nIterating through DataLoader:")
# for epoch in range(2):  # Example: 2 epochs
#     print(f"--- Epoch {epoch + 1} ---")
#     for i, batch in enumerate(data_loader):
#         features = batch["features"]
#         labels = batch["label"]
#         print(
#             f"Batch {i + 1}: Features shape: {features.shape}, Labels shape: {labels.shape}"
#         )
#         # print(f"Features:\n{features}")
#         # print(f"Labels:\n{labels}")
#         # In a real scenario, you would pass features and labels to your model
#         # outputs = model(features)
#         # loss = criterion(outputs, labels)
#         # loss.backward()
#         # optimizer.step()
