import os
from collections import defaultdict
from torch.utils.data import Dataset

class HandDataset(Dataset):
    def __init__(self, data_dir : str, file_list : str):
        self.data_dir = data_dir
        self.file_list = file_list
        self.dataset_samples = []

        # Prepare data
        self.prepare_data_list()
    
    def prepare_data_list(self,):
        with open(self.file_list, "r") as f:
            file_names = f.readlines()
        folders = sorted([x for x in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, x))])
        id2folder = defaultdict(list)
        for folder in folders:
            # check if there exists annotations
            contact_ann_file = os.path.join(self.data_dir, folder, "corresponding_contacts.json")
            if not os.path.exists(contact_ann_file):
                continue
            
            # save the id of the video (remove annotator mark)
            id = "_".join(folder.split("_")[:-1])
            id2folder[id].append(folder)

        for n in file_names:
            n = n.strip()
            self.dataset_samples.extend(id2folder[n])
    
    def __len__(self,):
        return len(self.dataset_samples)
    
    def __getitem__(self, index):
        folder_name = self.dataset_samples[index]
        return folder_name