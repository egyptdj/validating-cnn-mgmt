import os
import csv
import yaml
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torchio as tio
import monai
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score


def get_yaml(f):
    with open(f) as f:
        yaml_dict = yaml.safe_load(f)
    return yaml_dict


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def main(config_path):
    config = get_yaml(config_path)

    transforms = [
            tio.RescaleIntensity(out_min_max=(0.0,1.0)),
            tio.CropOrPad(config['image_size']),
            ]
    augment_transforms = [
            tio.RandomFlip(axes='LR'),
            tio.RandomAffine(scales=(0.9,1.2), degrees=10, isotropic=True, image_interpolation='nearest'),
            ]

    train_transform = tio.Compose(transforms+augment_transforms)
    test_transform = tio.Compose(transforms)

    df = pd.read_csv(os.path.join(config['data_dir'], 'train_labels.csv'), index_col=0, dtype='string')

    # start grid search validation
    for seed in config['sweeps']['seed']:
        for train_datasource in config['sweeps']['train_datasource']:
            merged_subject_list = [s for s in os.listdir(os.path.join(config['data_dir'], 'nifti_reg', 'train')) if not s in config['exclusion']]
            snuh_subject_list = [s for s in merged_subject_list if len(s)==8]
            rsnamiccai_subject_list = [s for s in merged_subject_list if len(s)==5]
            
            merged_label_list = [float(df.loc[df.index==subject]['MGMT_value']) for subject in merged_subject_list]
            snuh_label_list = [float(df.loc[df.index==subject]['MGMT_value']) for subject in snuh_subject_list]
            rsnamiccai_label_list = [float(df.loc[df.index==subject]['MGMT_value']) for subject in rsnamiccai_subject_list]

            if train_datasource == 'rsnamiccai':
                train_subject_list, val_subject_list, train_label_list, val_label_list = train_test_split(rsnamiccai_subject_list, rsnamiccai_label_list, test_size=0.1, random_state=seed, shuffle=True, stratify=rsnamiccai_label_list)
                test_subject_list, test_label_list = snuh_subject_list, snuh_label_list
            elif train_datasource == 'snuh':
                train_subject_list, val_subject_list, train_label_list, val_label_list = train_test_split(snuh_subject_list, snuh_label_list, test_size=0.1, random_state=seed, shuffle=True, stratify=snuh_label_list)
                test_subject_list, test_label_list = rsnamiccai_subject_list, rsnamiccai_label_list
            else:
                train_subject_list, test_subject_list, train_label_list, test_label_list = train_test_split(merged_subject_list, merged_label_list, test_size=0.1, random_state=seed, shuffle=True, stratify=merged_label_list)
                train_subject_list, val_subject_list, train_label_list, val_label_list = train_test_split(train_subject_list, train_label_list, test_size=0.1, random_state=seed, shuffle=True, stratify=train_label_list)

            for sequence in config['sweeps']['sequence']:
                for model_name in config['sweeps']['model']:
                    model_class = getattr(monai.networks.nets, model_name)
                    set_seed(seed)
                    expname = f'seed-{seed}_traindatasource-{train_datasource}_model-{model_name}_sequence-{"".join(sequence)}'
                    print('='*70)
                    print(expname)
                    print('='*70)
                    result_dir = os.path.join(config['result_dir'], expname)
                    os.makedirs(result_dir, exist_ok=True)
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                    criterion = nn.BCEWithLogitsLoss()

                    train_dataset = tio.SubjectsDataset([
                        tio.Subject(
                            FLAIR=tio.ScalarImage(os.path.join(config['data_dir'], 'nifti_reg', 'train', subject, 'flair.nii.gz')), 
                            T1w=tio.ScalarImage(os.path.join(config['data_dir'], 'nifti_reg', 'train', subject, 't1w.nii.gz')), 
                            T1wCE=tio.ScalarImage(os.path.join(config['data_dir'], 'nifti_reg', 'train', subject, 't1wce.nii.gz')), 
                            T2w=tio.ScalarImage(os.path.join(config['data_dir'], 'nifti_reg', 'train', subject, 't2w.nii.gz')), 
                            subject=subject, 
                            label=label) for subject, label in zip(train_subject_list, train_label_list)], transform=train_transform)

                    val_dataset = tio.SubjectsDataset([
                        tio.Subject(
                            FLAIR=tio.ScalarImage(os.path.join(config['data_dir'], 'nifti_reg', 'train', subject, 'flair.nii.gz')), 
                            T1w=tio.ScalarImage(os.path.join(config['data_dir'], 'nifti_reg', 'train', subject, 't1w.nii.gz')), 
                            T1wCE=tio.ScalarImage(os.path.join(config['data_dir'], 'nifti_reg', 'train', subject, 't1wce.nii.gz')), 
                            T2w=tio.ScalarImage(os.path.join(config['data_dir'], 'nifti_reg', 'train', subject, 't2w.nii.gz')), 
                            subject=subject, 
                            label=label) for subject, label in zip(val_subject_list, val_label_list)], transform=test_transform)

                    test_dataset = tio.SubjectsDataset([
                        tio.Subject(
                            FLAIR=tio.ScalarImage(os.path.join(config['data_dir'], 'nifti_reg', 'train', subject, 'flair.nii.gz')), 
                            T1w=tio.ScalarImage(os.path.join(config['data_dir'], 'nifti_reg', 'train', subject, 't1w.nii.gz')), 
                            T1wCE=tio.ScalarImage(os.path.join(config['data_dir'], 'nifti_reg', 'train', subject, 't1wce.nii.gz')), 
                            T2w=tio.ScalarImage(os.path.join(config['data_dir'], 'nifti_reg', 'train', subject, 't2w.nii.gz')), 
                            subject=subject, 
                            label=label) for subject, label in zip(test_subject_list, test_label_list)], transform=test_transform)
                        
                    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True, num_workers=config['num_workers'])
                    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=config['num_workers'])
                    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=config['num_workers'])

                    if model_class is monai.networks.nets.EfficientNetBN:
                        model = model_class('efficientnet-b0', in_channels=len(sequence), num_classes=1, spatial_dims=3, pretrained=False)
                    elif model_class is monai.networks.nets.DenseNet121:
                        model = model_class(in_channels=len(sequence), out_channels=1, spatial_dims=3, pretrained=False)
                    else:
                        model = model_class(in_channels=len(sequence), num_classes=1, spatial_dims=3, pretrained=False)
                    
                    model.to(device)
                    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
                    best_auroc = 0.0
                    best_loss = 1e9
                    best_accuracy = 0.0
                    stop_patience = 0

                    for epoch in range(100):
                        train_pred = []
                        train_prob = []
                        train_label = []
                        train_loss = []
                        model.train()
                        
                        for x in tqdm(train_dataloader, ncols=60, desc=str(epoch)):
                            volume = torch.cat([x[s][tio.DATA] for s in sequence], axis=1).to(device)
                            label = x['label'].to(device)

                            # forward
                            logit = model(volume)
                            loss = criterion(logit.squeeze(1), (label-0.01).abs())

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            train_loss.append(loss.detach().cpu().numpy())
                            train_pred.append(logit.sigmoid().round().detach().cpu().numpy())
                            train_prob.append(logit.sigmoid().detach().cpu().numpy())
                            train_label.append(label.detach().cpu().numpy())
                            torch.cuda.empty_cache()

                        train_loss = np.mean(train_loss)
                        train_auroc = roc_auc_score(np.concatenate(train_label), np.concatenate(train_prob))
                        train_accuracy = accuracy_score(np.concatenate(train_label), np.concatenate(train_pred))

                        val_prob = []
                        val_pred = []
                        val_label = []
                        val_loss = []
                        model.eval()
                        with torch.no_grad():
                            for x in val_dataloader:
                                volume = torch.cat([x[s][tio.DATA] for s in sequence], axis=1).to(device)
                                label = x['label'].to(device)

                                # forward
                                logit = model(volume)
                                loss = criterion(logit.squeeze(1), label)

                                val_loss.append(loss.detach().cpu().numpy())
                                val_pred.append(logit.sigmoid().round().detach().cpu().numpy())
                                val_prob.append(logit.sigmoid().detach().cpu().numpy())
                                val_label.append(label.detach().cpu().numpy())
                                torch.cuda.empty_cache()

                        val_loss = np.mean(val_loss)
                        val_auroc = roc_auc_score(np.concatenate(val_label), np.concatenate(val_prob))
                        val_accuracy = accuracy_score(np.concatenate(val_label), np.concatenate(val_pred))
                        val_precision = precision_score(np.concatenate(val_label), np.concatenate(val_pred))
                        val_recall = recall_score(np.concatenate(val_label), np.concatenate(val_pred))
                        scheduler.step(val_loss)
                        
                        with open(os.path.join(config['result_dir'], 'train_log.csv'), 'a') as f:
                            f.write(','.join([str(epoch), str(train_loss), str(train_accuracy), str(train_auroc), str(val_loss), str(val_accuracy), str(val_auroc)]))
                            f.write('\n')

                        if val_accuracy > best_accuracy:
                            stop_patience = 0
                            best_loss = val_loss
                            best_auroc = val_auroc
                            best_accuracy = val_accuracy
                            best_precision = val_precision
                            best_recall = val_recall
                            torch.save(model.state_dict(), os.path.join(result_dir, f'model_epoch{epoch}_bestacc.pth'))
                            print(f'>>>>>> best model saved with loss {best_loss:.4f} / auroc {best_auroc:.4f} / acc {best_accuracy:.4f}')
                            test_prob = []
                            test_pred = []
                            test_label = []
                            test_loss = []
                            model.eval()
                            with torch.no_grad():
                                for x in test_dataloader:
                                    volume = torch.cat([x[s][tio.DATA] for s in sequence], axis=1).to(device)
                                    label = x['label'].to(device)

                                    # forward
                                    logit = model(volume)
                                    loss = criterion(logit.squeeze(1), label)

                                    test_loss.append(loss.detach().cpu().numpy())
                                    test_pred.append(logit.sigmoid().round().detach().cpu().numpy())
                                    test_prob.append(logit.sigmoid().detach().cpu().numpy())
                                    test_label.append(label.detach().cpu().numpy())
                                    torch.cuda.empty_cache()

                            test_loss = np.mean(test_loss)
                            test_auroc = roc_auc_score(np.concatenate(test_label), np.concatenate(test_prob))
                            test_accuracy = accuracy_score(np.concatenate(test_label), np.concatenate(test_pred))
                            test_precision = precision_score(np.concatenate(test_label), np.concatenate(test_pred))
                            test_recall = recall_score(np.concatenate(test_label), np.concatenate(test_pred))
                            print(f'>>>>>> test performance: loss {test_loss:.4f} / auroc {test_auroc:.4f} / acc {test_accuracy:.4f}')
                            
                            result_dict = {
                                'seed': seed,
                                'model': repr(model_class),
                                'sequence': '-'.join(sequence),
                                'train_datasource': train_datasource,
                                'epoch': epoch,
                                'test_auroc': test_auroc,
                                'test_accuracy': test_accuracy,
                                'test_precision': test_precision,
                                'test_recall': test_recall,
                                'val_auroc': best_auroc,
                                'val_accuracy': best_accuracy,
                                'val_precision': best_precision,
                                'val_recall': best_recall,
                            }

                            with open(os.path.join(result_dir, 'result.csv'), 'w') as f:
                                w = csv.writer(f)
                                w.writerow(result_dict.keys())
                                w.writerow(result_dict.values())
                        else:
                            stop_patience += 1

                        if stop_patience >= 15:
                            print('='*70)
                            print(f'>>>>>> STOPPING AT EPOCH {epoch}')
                            print('='*70)
                            break


if __name__ == '__main__':
    main('config.yaml')
    exit(0)