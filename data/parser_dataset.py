import multiprocessing as mp
from data import all_atom_parse as aap
from data.all_atom_parse import dump_structure_data
import glob
import os
from tqdm import tqdm
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import gzip



def process_file(raw_data):
    try:
        # pdb = raw_data.split('/')[-1].split('.')[0]
        # assembly_path = '/ssd/dataset/pdb/train/'+pdb+'.cif.gz'
        # with gzip.open(raw_data, 'rt') as f:
        #         mmcif_dict = MMCIF2Dict(f)
        # method = mmcif_dict['_exptl.method'][0]

        # if method not in ['X-RAY DIFFRACTION', 'ELECTRON MICROSCOPY']:
        #     print(raw_data,'method:',method)
        #     #remove assembly file
        #     os.remove(assembly_path)
        #     return None
        
        # if '_em_3d_reconstruction.resolution' in mmcif_dict:
        #     resolution = mmcif_dict['_em_3d_reconstruction.resolution'][0]
        # # Try X-ray resolution fields
        # elif '_refine.ls_d_res_high' in mmcif_dict:
        #     resolution = mmcif_dict['_refine.ls_d_res_high'][0]
        # elif '_reflns.d_resolution_high' in mmcif_dict:
        #     resolution = mmcif_dict['_reflns.d_resolution_high'][0]

        # if float(resolution) > 3.5:
        #     print(raw_data,'resolution:',resolution)
        #     #remove assembly file
        #     os.remove(assembly_path)
        #     return None

        data = aap.parse_or_load_mmcif(raw_data)
        if 'train' in raw_data:
            save_path = raw_data.replace('train', 'train_parsed')
        elif 'test' in raw_data:
            save_path = raw_data.replace('test', 'test_parsed')
        elif 'validation' in raw_data:
            save_path = raw_data.replace('validation', 'validation_parsed')

        pdb = save_path.split('/')[-1].split('.')[0]
        path_parts = save_path.split('/')[:-1]  # Split and remove last part
        base_path = os.path.join(*path_parts)   # Join the parts back together
        final_path = '/'+os.path.join(base_path, pdb[1:3], f"{pdb}.npz")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        dump_structure_data(data, final_path)
    except Exception as e:
        print(f"Error processing {raw_data}: {str(e)}")

def main():
    data_path = '/ssd/dataset/pdb/train/'
    raw_data_list = glob.glob(data_path + '*.cif.gz')

    # Determine the number of CPU cores to use
    num_cores = mp.cpu_count()
    
    # Create a pool of worker processes
    with mp.Pool(processes=num_cores) as pool:
        # Use tqdm to create a progress bar
        for _ in tqdm(pool.imap_unordered(process_file, raw_data_list), total=len(raw_data_list)):
            pass

if __name__ == '__main__':
    main()