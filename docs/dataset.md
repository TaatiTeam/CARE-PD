# CARE-PD: A Comprehensive 3D Gait Dataset for Parkinson's Disease

## Overview

CARE-PD is the largest publicly available archive of 3D mesh gait data for Parkinson's Disease (PD) and the first to include data collected across multiple sites. The dataset aggregates 9 cohorts from 8 clinical sites, including 363 participants spanning a range of disease severity. All recordings—whether from RGB video or motion capture—are unified into anonymized SMPL body gait meshes through a curated harmonization pipeline.

This dataset enables two main benchmarks:
1. **Supervised clinical score prediction**: Estimating UPDRS gait scores from 3D meshes
2. **Unsupervised motion pretext tasks**: Parkinsonian gait representation learning

## Dataset Contents

CARE-PD consists of 9 harmonized datasets:

1. **3DGait** - Clinical gait recordings with UPDRS scores
2. **BMCLab** - Gait recordings with medication status and UPDRS scores
3. **DNE** - Contains healthy, Parkinson's, and other neurological conditions
4. **E-LC** - Medication status (on/off) with freezing events
5. **KUL-DT-T** - Freezer/non-freezer classification data
6. **PD-GaM** - Clinical gait recordings with UPDRS scores
7. **T-SDU** - General walking recordings 
8. **T-SDU-PD** - PD patient walking with UPDRS scores
9. **T-LTC** - Additional walking recordings

## Data Structure

Each dataset is provided in a standardized format:

```
{
    "anonymized_subject_id": {
        "anonymized_walk_id": {
            "pose": array,      # SMPL pose parameters (shape varies by dataset)
            "trans": array,     # Translation data
            "beta": array,      # Body shape parameters (zeros for privacy)
            "fps": int,         # Frames per second (standardized)
            "UPDRS_GAIT": int,  # Clinical score (0-3) or None if unavailable
            "medication": str,  # Medication status or None if unavailable
            "other": str        # Additional labels or None if unavailable
        }
    }
}
```

## Getting Started

### Loading Data

We provide utility functions for loading the 3D skeletal data in the repository.

TO DO (VIDA Code)

### Data Visualization

We provide utility functions for visualizing the 3D skeletal data in the repository.
```
python utility/viz_seqs.py -n assets/datasets/h36m/BMCLab/h36m_3d_world2cam2img_sideright_floorXZZplus_30f_or_longer.npz  -f h36m -p 2d

python utility/viz_seqs.py -n assets/datasets/h36m/BMCLab/h36m_3d_world_floorXZZplus_30f_or_longer.npz  -f h36m
```

## Benchmarks

CARE-PD includes data splits to test generalization:
1. 6-Fold (split per subject)
2. Leave-one-subject-out
3. Fixed train-test splits (split per subject)

The former two are only provided for the supervised clinical score prediction task.

## Terms of Use

By accessing and using this database (the "Database"), users ("Users") acknowledge and agree to comply with the following conditions:

### 1. License and Attribution
- The Database is publicly released under a Creative Commons Attribution-NonCommercial (CC BY-NC 4.0) license.
- Users must provide appropriate attribution by citing the Datbase and the original publications associated with each dataset accessed from the Database. 

### 2. Data Privacy and Ethics
- Users must not attempt to identify, contact, or otherwise compromise the anonymity of any individuals whose data is included in the Database.
- All use of the data must comply with applicable ethical guidelines and legal regulations, including privacy laws (e.g., GDPR, HIPAA, PIPEDA).

### 3. Data Handling and Security
- Users must maintain appropriate data security measures to prevent unauthorized access, sharing, or use of the data.
- Users are encouraged, but not required, to direct third parties to the original Database URL rather than re-hosting the data.

### 4. Intellectual Property Notice
- Copyright and other rights remain with the original data providers.

### 5. Disclaimer of Warranty
- Use of the Database is subject to Section 5 (Disclaimer of Warranties and Limitation of Liability) of the CC BY-NC 4.0 licence.

By using the Database, Users expressly acknowledge and agree to abide by these Terms of Use.

## Citation

If you use CARE-PD in your research, please cite:

```
[Citation information to be provided upon publication]
```

## Acknowledgments

We thank all participating research institutions and subjects who made this dataset possible.

## Contact

For questions, issues, or additional information, please contact:
[Contact information to be provided]
