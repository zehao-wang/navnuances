# Execute VLN-DUET Model on the NavNuances Dataset

### 1. Download Necessary Data
* Download the ```R2R/features```, ```R2R/connectivity``` and ```R2R/trained_models``` folders from [link](https://www.dropbox.com/sh/u3lhng7t2gq36td/AABAIdFnJxhhCg2ItpAhMtUBa?dl=0) and place them under the ```VLN-DUET/datasets/R2R``` directory.

* Download ```R2R_train_enc.json``` from [link](https://www.dropbox.com/scl/fo/4iaw2ii2z2iupu0yn4tqh/AN7bYSotG-zBLzM11i2d0H8/R2R/annotations?dl=0&rlkey=88khaszmvhybxleyv0a9bulyn&subfolder_nav_tracking=1) and put it in ```VLN-DUET/datasets/R2R/annotations``` to avoid error during training file checks.

* Link or put all the files of NavNuances dataset into ```VLN-DUET/datasets/R2R/annotations```, and place R2R standard validation unseen split ```R2R_val_unseen.json``` to the same directory as well.

### 2. Modify Validation Code

Including NavNuances in existing methods is simple, we provide a version under ```baselines/VLN-DUET```. The main changes compared to official repo are:
1. Modify the instruction parser in ```map_nav_src/r2r/data_utils.py``` file to enable online tokenization.

2. Include NavNuances split names in ```map_nav_src/r2r/main_nav.py``` by making the following changes:

```python
61 - val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']
62 - if args.dataset == 'r4r' and (not args.test):
.       val_env_names[-1] == 'val_unseen_sampled'
.    
.     if args.submit and args.dataset != 'r4r':
68 -    val_env_names.append('test')
------------------------------------
61 + val_env_names = ['DC', 'LR', 'NU', 'RR', 'VM', 'val_unseen']
62 + # if args.dataset == 'r4r' and (not args.test):
.    #     val_env_names[-1] == 'val_unseen_sampled'
.   
.    # if args.submit and args.dataset != 'r4r':
68 + #     val_env_names.append('test')
```
```python
73 - is_test=is_test
73 + is_test=is_test, tokenizer_obj=tok
```
```python
251 - if 'test' not in env_name:
251 + if env_name not in ['DC', 'LR', 'NU', 'RR', 'VM', 'test']: 
```

### 3. Generate Prediction Results
```bash
cd VLN-DUET/map_nav_src
sh scripts/run_eval_navnuances.sh
```
The predicted trajectories are stored under ```VLN-DUET/datasets/R2R/exprs_map```