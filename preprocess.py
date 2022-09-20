import os
import ants
import shutil
import argparse
import dicom2nifti
from tqdm import tqdm


SEQUENCES = ['T1w', 'T1wCE', 'T2w', 'FLAIR']


def convert_to_nifti(source_dir, target_dir, split):
    # dicom to nifti
    for subject in tqdm(os.listdir(os.path.join(source_dir, split)), desc=split, ncols=60):
        for seq in SEQUENCES:
            if len(os.listdir(os.path.join(source_dir, split, subject, seq))) < 4:
                continue
            os.makedirs(os.path.join(target_dir, split, subject), exist_ok=True)
            dicom2nifti.convert_directory(os.path.join(source_dir, split, subject, seq), os.path.join(target_dir, split, subject), compression=True)
        [os.rename(os.path.join(target_dir, split, subject, f), os.path.join(target_dir, split, subject, f.split('_')[-1])) for f in os.listdir(os.path.join(target_dir, split, subject))]


def registration(source_dir, target_dir, split, template_sequence='T1w', type_of_transform='Rigid', template_image='mni'):
    fixed = ants.image_read(ants.get_data(template_image))
    for i, subject in enumerate(tqdm(os.listdir(os.path.join(source_dir, split)), desc=split, ncols=60)):
        os.makedirs(os.path.join(target_dir, split, subject), exist_ok=True)
        for seq in SEQUENCES:
            try:
                moving = ants.image_read(os.path.join(source_dir, split, subject, f'{seq.lower()}.nii.gz'), reorient=True)
                if seq==template_sequence:
                    _t = ants.registration(fixed=fixed, moving=moving, type_of_transform=type_of_transform)
                    fwdtransforms = _t['fwdtransforms']
                    transformed = _t['warpedmovout']
                else:
                    transformed = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=fwdtransforms)
            except Exception as e:
                continue

            # save to file
            transformed.to_file(os.path.join(target_dir, split, subject, f'{seq.lower()}.nii.gz'))

        if len(os.listdir(os.path.join(target_dir, split, subject)))<4:
            print(f'removing subject {subject} with {len(os.listdir(os.path.join(target_dir, split, subject)))} sequences')
            shutil.rmtree(os.path.join(target_dir, split, subject))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default='rsna-miccai-brain-tumor-radiogenomic-classification')
    parser.add_argument('--target_dir', type=str, default='data')
    parser.add_argument('--split', type=str, default='train')
    argv = parser.parse_args()

    shutil.copyfile(os.path.join(argv.source_dir, 'train_labels.csv'), os.path.join(argv.target_dir, 'train_labels.csv'))

    print('>>> dicom to nifti conversion')
    convert_to_nifti(os.path.join(argv.source_dir), os.path.join(argv.target_dir, 'nifti'), argv.split)
    
    print('>>> registration')
    registration(os.path.join(argv.target_dir, 'nifti'), os.path.join(argv.target_dir, 'nifti_reg'), argv.split)
    
    exit(0)