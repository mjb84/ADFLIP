import torch
import glob
import os
import traceback
import numpy as np
import pandas as pd
import torch.utils
import torch.utils.data
import random

from data.all_atom_parse import residue_tokens
from data import all_atom_parse as aap
from data.all_atom_parse import StructureData
from data.utils import TimeSortedCacheRBT


def propagate_mask_vectorized(mask, index):
    # Ensure inputs are tensors
    mask = torch.as_tensor(mask, dtype=torch.bool)
    index = torch.as_tensor(index)
    
    # Get unique indices and their inverse mapping
    unique_indices, inverse_indices = torch.unique(index, return_inverse=True)
    
    # Create a tensor to hold the "any True" status for each unique index
    any_true = torch.zeros_like(unique_indices, dtype=torch.bool)
    
    # Use scatter_reduce to check if any mask is True for each unique index
    any_true.scatter_reduce_(0, inverse_indices, mask, reduce='amax')
    
    # Expand the any_true tensor to match the original shape
    return any_true[inverse_indices]




class AllAtomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg,
        dataset_type: str = "train",
        use_cache: bool = True,
        cache_length: int = -1,
        test_time=0.0,
    ):
        # if dataset_type == "train":
        self.max_num_residues = cfg.max_num_residues
        self.max_num_atoms = cfg.max_num_atoms
        # else:
        #     self.max_num_residues = 100000
        #     self.max_num_atoms = 100000

        self.cut_position_type = cfg.cut_position_type
        self.random_gen = np.random.default_rng(cfg.random_seed)
        self.dataset_type = dataset_type
        paths = {
            "train": cfg.train_path,
            "valid": cfg.valid_path,
            "test": cfg.test_path,
        }

        self.structs = (
            glob.glob(f"{paths[dataset_type]}/*.pdb")
            + glob.glob(f"{paths[dataset_type]}/*.cif")
            + glob.glob(f"{paths[dataset_type]}/*.npz")
        )
        if len(self.structs) == 0:
            self.structs = (
                glob.glob(f"{paths[dataset_type]}/**/*.pdb")
                + glob.glob(f"{paths[dataset_type]}/**/*.cif")
                + glob.glob(f"{paths[dataset_type]}/**/*.npz")
            )
        self.test_time = test_time
        self.use_cache = use_cache

        if self.use_cache:
            self.cache_data = TimeSortedCacheRBT(
                capacity=cache_length, parse_fn=aap.parse_or_load_mmcif
            )

    def __len__(self):
        return len(self.structs)

    def __getitem__(self, idx) -> aap.StructureData:
        try:
            if self.use_cache:
                data = self.cache_data.get(self.structs[idx])
            else:
                data = aap.parse_or_load_mmcif(self.structs[idx])
            if isinstance(data, tuple):
                data = data[1]
            num_residues = data.num_residues()
        except Exception as e:
            print(f"Failed to parse {self.structs[idx]}")
            return None

        if num_residues > self.max_num_residues:
            # Here can do all kinds of complicated logic about finding ligands
            # that are closest to protein residues etc
            # For now, just pick a random protein residue
            if self.cut_position_type == "protein":
                random_position = self.random_gen.choice(
                    data.position[data.is_protein], axis=0
                )
            elif self.cut_position_type == "ligand":
                if data.is_protein.sum() == data.is_protein.shape[0]: # no ligand
                    random_position = self.random_gen.choice(
                        data.position, axis=0
                    )
                    # print("No ligand found, path", self.structs[idx])
                else:
                    random_position = self.random_gen.choice(
                    data.position[~data.is_protein], axis=0
                )

            data = aap.get_closest_n_residues(
                data, random_position, n=self.max_num_residues
            )

        if len(data) > self.max_num_atoms:
            if data.is_protein.sum() == 0:
                print("No protein found, path", self.structs[idx])
                return None
            else:
                random_position = self.random_gen.choice(
                    data.position[data.is_protein], axis=0
                )
                data = aap.get_closest_n_atoms(
                    data,
                    random_position,
                    n=self.max_num_atoms,
                )

        # I only know how to do this in torch...
        # Mask proteins without 4 backbone atoms
        num_residues = np.max(data.residue_index) + 1
        num_backbones = torch.zeros(num_residues)
        num_backbones.index_add_(
            0,
            torch.from_numpy(data.residue_index),
            torch.from_numpy(data.is_backbone).float(),
        )
        full_backbone = (num_backbones == 4).numpy()
        full_backbone_atoms = np.copy(data.not_pad_mask)
        full_backbone_atoms = full_backbone[data.residue_index]
        full_backbone_mask = (full_backbone_atoms) | (~data.is_protein)
        data = aap.slice_structure_data(data, full_backbone_mask)

        if data.is_protein.sum() == 0:
            print("No protein found, path", self.structs[idx])
            return None
        data = self.interact_residue(data)

        if self.dataset_type == "train":
            return self.corrupt(data)
        else:
            return self.corrupt(data, time=np.array([self.test_time]))

    def corrupt(self, data: aap.StructureData, time=None) -> aap.StructureData:
        # Corrupt the data in some way
        raw_data = data.__dict__
        batch_dict = data.__dict__
        num_residue_tokens = batch_dict["residue_index"].max() + 1
        if time is not None:
            time_step = time
        else:
            time_step = np.random.uniform(0, 1, 1)
        u = np.random.uniform(0, 1, num_residue_tokens)
        target_mask = u < (1.0 - time_step)  # True -> mask

        noisy_residue_token = np.copy(batch_dict["residue_token"])
        noisy_residue_token[
            target_mask[batch_dict["residue_index"]] & batch_dict["is_protein"]
        ] = residue_tokens["<MASK>"]
        batch_dict["noisy_residue_token"] = noisy_residue_token

        mask_sidechain = np.ones_like(batch_dict["residue_token"]).astype(bool)
        mask_sidechain[~batch_dict["is_backbone"] & batch_dict["is_protein"]] = (
            ~target_mask[
                batch_dict["residue_index"][
                    ~batch_dict["is_backbone"] & batch_dict["is_protein"]
                ]
            ]
        )
        # print(mask_sidechain.shape,raw_data['is_protein'].shape)

        for name, data in batch_dict.items():
            if name == "noisy_residue_token":
                batch_dict[name] = noisy_residue_token[mask_sidechain]
            elif name == "time_step":
                batch_dict[name] = time_step
            else:
                batch_dict[name] = data[mask_sidechain]

        return StructureData(**batch_dict)

    def interact_residue(self,data):
            protein_pos = np.expand_dims(data.position[data.is_protein],axis=1)
            non_protein_pos = np.expand_dims(data.position[~data.is_protein],axis=0)
            ion_pos = np.expand_dims(data.position[data.is_ion],axis=0)
            nucleotide_pos = np.expand_dims(data.position[data.is_nucleotide],axis=0)
            molecule_pos = np.expand_dims(data.position[~data.is_protein&~data.is_ion&~data.is_nucleotide],axis=0)

            dist_non_protein = np.sqrt((np.sum((protein_pos-non_protein_pos)**2,axis=-1)))
            interact_non_protein = np.any(dist_non_protein<5,axis=1)
            dist_ion = np.sqrt((np.sum((protein_pos-ion_pos)**2,axis=-1)))
            interact_ion = np.any(dist_ion<5,axis=1)
            dist_nucleotide = np.sqrt((np.sum((protein_pos-nucleotide_pos)**2,axis=-1)))
            interact_nucleotide = np.any(dist_nucleotide<5,axis=1)
            dist_molecule = np.sqrt((np.sum((protein_pos-molecule_pos)**2,axis=-1)))
            interact_molecule = np.any(dist_molecule<5,axis=1)


            interact_non_protein_res = np.zeros(data.residue_index.shape[0],dtype=bool)
            interact_non_protein_res[data.is_protein] = propagate_mask_vectorized(interact_non_protein,data.residue_index[data.is_protein])
            interact_ion_res = np.zeros(data.residue_index.shape[0],dtype=bool)
            interact_ion_res[data.is_protein] = propagate_mask_vectorized(interact_ion,data.residue_index[data.is_protein])
            interact_nucleotide_res = np.zeros(data.residue_index.shape[0],dtype=bool)
            interact_nucleotide_res[data.is_protein] = propagate_mask_vectorized(interact_nucleotide,data.residue_index[data.is_protein])
            interact_molecule_res = np.zeros(data.residue_index.shape[0],dtype=bool)
            interact_molecule_res[data.is_protein] = propagate_mask_vectorized(interact_molecule,data.residue_index[data.is_protein])


            data.interact_non_protein_res = interact_non_protein_res
            data.interact_ion_res = interact_ion_res
            data.interact_nucleotide_res = interact_nucleotide_res
            data.interact_molecule_res = interact_molecule_res

            return data

class PlinderDataset(AllAtomDataset):
    def __init__(
        self,
        cfg,
        dataset_type: str = "train",
        use_cache: bool = True,
        cache_length: int = -1,
        test_time=0.0,
    ):
        super().__init__(cfg, dataset_type, use_cache, cache_length, test_time)
        self.base_dir = os.path.split(cfg.train_path)[0]
        self.df = pd.read_csv(os.path.join(self.base_dir, dataset_type + ".csv"))
        paths = {
            "train": cfg.train_path,
            "valid": cfg.valid_path,
            "test": cfg.test_path,
        }


        if self.use_cache:
            self.structs = [
                os.path.join(
                    paths[dataset_type],
                    x["system_id"][1:3],
                    x["system_id"] + ".npz"
                ) for _, x in self.df.iterrows()
            ]
        else:
            self.structs = [
                os.path.join(
                    paths[dataset_type],
                    x["system_id"][1:3],
                    x["system_id"] + ".cif"
                ) for _, x in self.df.iterrows()
            ]
        self.clusters = self.df.cluster.unique()

    def __len__(self) -> int:
        return len(self.structs)

    def __getitem__(self, idx) -> StructureData:
        cluster = self.clusters[idx % len(self.clusters)]
        sample = self.df[self.df.cluster == cluster].sample(
            random_state=self.random_gen
        )
        index = int(sample.index[0])
        return super().__getitem__(index)

class Cluster_AllAtomDataset(AllAtomDataset):
    def __init__(
        self,
        cfg,
        dataset_type: str = "train",
        use_cache: bool = True,
        cache_length: int = -1,
        test_time=0.0,
    ):
        super().__init__(cfg, dataset_type, use_cache, cache_length, test_time)
        self.base_dir = os.path.split(cfg.train_path)[0]

        self.cluster_dict = torch.load(os.path.join(self.base_dir, 'train_cluster_dict.pt'))
        self.all_pdb = []
        for cluster in self.cluster_dict.values():
            self.all_pdb.extend(cluster)

        paths = {
            "train": cfg.train_path,
            "valid": cfg.valid_path,
            "test": cfg.test_path,
        }

        self.data_dir = os.path.join(self.base_dir,paths[dataset_type])

        if self.use_cache:
            self.structs = [
                os.path.join(
                    paths[dataset_type],
                    x[1:3],
                    x + ".npz"
                ) for x in self.all_pdb
            ]
        else:
            self.structs = [
                os.path.join(
                    paths[dataset_type],
                    x["system_id"][1:3],
                    x["system_id"] + ".cif"
                ) for _, x in self.df.iterrows()
            ]

    def __len__(self) -> int:
        return len(self.cluster_dict.keys())

    def __getitem__(self, idx) -> StructureData:
        cluster = self.cluster_dict[idx]
        sample = random.choice(cluster)
        if self.use_cache:
            sample_pdb_path = os.path.join(self.data_dir,sample[1:3],sample + ".npz")
        else:
            sample_pdb_path = os.path.join(''.join(self.structs[0].split('/')[:-2]),sample[1:3],sample + ".cif")
        index = self.structs.index(sample_pdb_path)
        return super().__getitem__(index)


if __name__ == "__main__":
    from functools import partial

    # class Config:
    #     train_path = "/ceph.groups/scheres_grp/kjamali/Kai_CATH42/test"
    #     test_path = "null"
    #     valid_path = "null"
    #     max_num_residues = 512
    #     random_seed = 42

    # dataset = AllAtomDataset(Config(), use_cache=True)

    class Config:
        train_path = "/ceph/scheres_grp/kyi/Documents/dataset/ligand_pdb/train_parsed"
        test_path = "null"
        valid_path = "null"
        max_num_atoms = 4096
        max_num_residues = 512
        random_seed = 42
        cut_position_type = "ligand"
        use_cathe = True

    dataset = Cluster_AllAtomDataset(Config())

    for i in range(10):
        print(dataset[i])

    # dataloader = torch.utils.data.DataLoader(
    #     dataset, 2, collate_fn=partial(aap.batch_structure_data_list, to_torch=True)
    # )

    # for sample in dataloader:
    #     print(sample)
    #     break
