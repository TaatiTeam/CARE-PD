import joblib
import pkg_resources
import wandb
import datetime

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from data.dataloaders import *
from model.motion_encoder import MotionEncoder
from model.backbone_loader import load_pretrained_backbone, count_parameters, load_pretrained_weights
from train import train_model, validate_model, final_test
from utility import utils
from const import path
from const import const
from learning.utils import log_cfm_to_wandb

from torch import nn


def log_results(rep, rep012, confusion, rep_name, rep012_name, conf_name, out_p):
    print("Complete classification report:")
    print(rep)
    print("Classification report for UPDRS=0,1,2:")
    print(rep012)
    assert confusion.shape[0] == confusion.shape[1], f"confusion.shape[0] == confusion.shape[1] failed {confusion.shape}"
    num_classes = confusion.shape[0]
    fig, ax = plt.subplots(figsize=(10, 10)) 
    sns.heatmap(confusion, annot=True, ax=ax, cmap="Blues", fmt='g', annot_kws={"size": 26}, cbar=False)
    ax.set_xlabel('Predicted labels', fontsize=28)
    ax.set_ylabel('True labels', fontsize=28)
    ax.set_title('Confusion Matrix', fontsize=30)
    ax.xaxis.set_ticklabels([f'class {i}' for i in range(num_classes)], fontsize=22)  # Modify class names as needed
    ax.yaxis.set_ticklabels([f'class {i}' for i in range(num_classes)], fontsize=22)
    # Save the figure
    plt.savefig(os.path.join(out_p, conf_name))
    wandb.log({f"final_test_cfm": wandb.Image(fig)})
    plt.close(fig)
    with open(os.path.join(out_p, rep_name), "w") as text_file:
        text_file.write(rep)
    with open(os.path.join(out_p, rep012_name), "w") as text_file:
        text_file.write(rep012)
    
    artifact = wandb.Artifact(f'confusion_matrices', type='image-results')
    artifact.add_file(os.path.join(out_p, conf_name))
    wandb.log_artifact(artifact)
    
    artifact = wandb.Artifact('reports', type='txtfile-results')
    artifact.add_file(os.path.join(out_p, rep_name))
    wandb.log_artifact(artifact)

    artifact = wandb.Artifact('reports012', type='txtfile-results')
    artifact.add_file(os.path.join(out_p, rep012_name))
    wandb.log_artifact(artifact)

def test__hypertune(params, best_params, new_params, splits, backbone_name, clarification_str, model_checkpoint_clarification_str, exp_path, device): 
    out_path = os.path.join(exp_path, clarification_str)
    params['clarification'] = clarification_str
    params['model_checkpoint_clarification_str'] = model_checkpoint_clarification_str
    tags = [
        f"{backbone_name}",
        f"dataset:{params['dataset']}",
        f"cross_dataset:{bool(params['cross_dataset_test'])}"
        f"train_mode:{params['train_mode']}",
        f"num_folds:{params['num_folds']}",
        f"run_num:{params['this_run_num']}",
        f"view:{''.join(params['views'])}",
        clarification_str
    ]
    wandb.init(project='Final_MotionEval_test',
            group=f"{params['experiment_name']}_test",
            job_type=f"test_crossdataset{bool(params['cross_dataset_test'])}",
            name=f"{backbone_name}_{clarification_str}_{''.join(params['views'])}",
            tags=tags,
            settings=wandb.Settings(start_method='fork'))
    wandb.config.update(params)
    installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}
    wandb.config.update({'installed_packages': installed_packages})
    wandb.config.update({'new_params': new_params})
    wandb.config.update({'best_params': best_params})
    
    total_outs_last, total_gts, total_logits, total_states, total_video_names = [], [], [], [], []
    for fold, (train_dataset_fn, test_dataset_fn, class_weights) in enumerate(splits, start=1):
        checkpoint_root_path = os.path.join(exp_path, 'models', params['model_checkpoint_clarification_str'], f"fold{fold}")
        last_ckpt_path = os.path.join(checkpoint_root_path, 'latest_epoch.pth.tr')
        res_dir = os.path.join(out_path, 'results')
        res_json_dir = os.path.join(res_dir, 'results_last_fold{}.json'.format(fold))
        logits_dir = os.path.join(out_path, 'logits')
        logits_json_dir = os.path.join(logits_dir, 'logits_last_fold{}.json'.format(fold))
        total_fold_results_dir = os.path.join(out_path, f'total_results_fold{fold}.pkl')
        if os.path.exists(total_fold_results_dir) and not params['overwrite_results']:
            with open(total_fold_results_dir, 'rb') as file:
                total_results = pickle.load(file)
            total_video_names = total_results['total_video_names'].tolist()
            total_outs_last = total_results['total_outs_last'].tolist()
            total_gts = total_results['total_gts'].tolist()
            total_states = total_results['total_states'].tolist()
            total_logits = []
            print(f"Fold {fold} already computed as {total_fold_results_dir} exists, moving on....")
            continue

        start_time = datetime.datetime.now()

        model_backbone = load_pretrained_backbone(params, backbone_name)
        model = MotionEncoder(backbone=model_backbone,
                                params=params,
                                num_classes=params['num_classes'],
                                train_mode=params['train_mode'])
        model = model.to(device)
        if fold == 1:
            model_params = count_parameters(model)
            print(f"[INFO] Model has {model_params} parameters.")
        
        if not params['pretrained'] and not os.path.exists(last_ckpt_path):
            print(f"{last_ckpt_path} does not exists so the model will be trained...")
            if torch.cuda.device_count() > 1:
                print("Using", torch.cuda.device_count(), "GPUs!")
                model = nn.DataParallel(model)

            train_model(params, class_weights, train_dataset_fn, None, model, fold, backbone_name)
            
            _, _, _, cfm_train = validate_model(model, train_dataset_fn, params, class_weights)
            log_cfm_to_wandb(cfm_train, fold, params['num_classes'], kind='train') # So we can see what the model is doing at the end of training
        
        load_pretrained_weights(model, checkpoint=torch.load(last_ckpt_path)['model'])
        model.cuda()
        print(f"Performing final test the model {clarification_str}...")
        outs_last, gts, logits, states, video_names = final_test(model, test_dataset_fn, params)
        total_outs_last.extend(outs_last)
        total_gts.extend(gts)
        total_states.extend(states)
        total_video_names.extend(video_names)
        total_logits.extend(logits)
        print(f'fold # of test samples: {len(video_names)}')
        print(f'current sum # of test samples: {len(total_video_names)}')
        attributes = [total_outs_last, total_gts, total_video_names]
        names = ['predicted_classes', 'true_labels', 'video_names']
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        utils.save_json(res_json_dir, attributes, names)

        attributes = [total_logits, total_gts, total_video_names]
        names = ['predicted_logits', 'true_labels', 'video_names']
        if not os.path.exists(logits_dir):
            os.makedirs(logits_dir)
        utils.save_json(logits_json_dir, attributes, names)
        
        res = pd.DataFrame({'total_video_names': total_video_names, 'total_outs_last': total_outs_last, 'total_gts':total_gts, 'total_states':total_states})
        with open(total_fold_results_dir, 'wb') as file:
            pickle.dump(res, file)

        end_time = datetime.datetime.now()
        duration = end_time - start_time
        print(f"Fold {fold} run time:", duration)
    
    res = pd.DataFrame({'total_video_names': total_video_names, 'total_outs_last': total_outs_last, 'total_gts':total_gts})
    with open(os.path.join(out_path, 'final_results.pkl'), 'wb') as file:
        pickle.dump(res, file)
    
    #================================LAST REPORTS=============================
    print(clarification_str)
    rep_last_final = classification_report(total_gts, total_outs_last)
    rep_last_final_just012 = classification_report(total_gts, total_outs_last, labels=[l for l in const.LABELS_INCLUDED_IN_F1_CALCULATION if l in total_gts])
    confusion_last_final = confusion_matrix(total_gts, total_outs_last)
    log_results(rep_last_final, rep_last_final_just012, confusion_last_final, 'last_report_allfolds.txt', 'last_report_allfolds_just012updrs.txt', 'last_confusion_matrix_allfolds.png', out_path)
    cls_rep = classification_report(total_gts, total_outs_last, labels=[l for l in const.LABELS_INCLUDED_IN_F1_CALCULATION if l in total_gts], output_dict=True)
    wandb.log({
        'macro avg precision': cls_rep['macro avg']['precision'],
        'macro avg recall': cls_rep['macro avg']['recall'],
        'macro avg f1-score': cls_rep['macro avg']['f1-score'],
        'weighted avg precision': cls_rep['weighted avg']['precision'],
        'weighted avg recall': cls_rep['weighted avg']['recall'],
        'weighted avg f1-score': cls_rep['weighted avg']['f1-score'],
    })
    wandb.finish()
    
    
def test_combine_view_predictions(path_back, path_side, last_fold, out_path, clarification_str, backbone_name, params):
    print('Combined results for')
    print(f"view0: {path_back}")
    print(f"view1: {path_side}")
    print(f"combined: {out_path}")
    os.makedirs(out_path, exist_ok=True)
    params['views'] = ['combined']
    tags = [
        f"{backbone_name}",
        f"dataset:{params['dataset']}",
        f"cross_dataset:{bool(params['cross_dataset_test'])}"
        f"train_mode:{params['train_mode']}",
        f"num_folds:{params['num_folds']}",
        f"run_num:{params['this_run_num']}",
        f"view:combined",
    ]
    wandb.init(project='Final_MotionEval_test',
            group='combine_views',
            job_type=f"test_crossdataset{bool(params['cross_dataset_test'])}",
            name=f"Combinedview_{backbone_name}_{clarification_str}",
            tags=tags,
            settings=wandb.Settings(start_method='fork'))
    wandb.config.update(params)
    
    saved_path = os.path.join(path_back, 'logits', 'logits_last_fold{}.json'.format(last_fold))
    with open(saved_path, 'r') as f:
        logits_v0 = json.load(f)
    saved_path = os.path.join(path_side, 'logits', 'logits_last_fold{}.json'.format(last_fold))
    with open(saved_path, 'r') as f:
        logits_v1 = json.load(f)
    logits_map_0 = utils.build_logit_map(logits_v0)
    logits_map_1 = utils.build_logit_map(logits_v1)
    # Check for missing keys
    diff_keys_0 = set(logits_map_0.keys()) - set(logits_map_1.keys())
    diff_keys_1 = set(logits_map_1.keys()) - set(logits_map_0.keys())
    if diff_keys_0 or diff_keys_1:
        print("🔍 Keys in logits_map_0 but not in logits_map_1:", diff_keys_0)
        print("🔍 Keys in logits_map_1 but not in logits_map_0:", diff_keys_1)
    else:
        print("✅ Both maps have the same keys.")
    # Combine predictions
    avg_logits = []
    predicted_labels = []
    video_names = []
    true_labels = []
    common_keys = set(logits_map_0.keys()) & set(logits_map_1.keys())
    for base_name in sorted(common_keys):
        l0 = logits_map_0[base_name]
        l1 = logits_map_1[base_name]
        avg = (l0 + l1) / 2
        if params['prefer_right']:
            candidates = np.flatnonzero(avg == np.max(avg))
            if len(candidates) == 1:
                pred = candidates[0]
            else:
                # tie happened
                pred = candidates[np.argmax(l1[candidates])]
            predicted_labels.append(pred)
        else:
            avg_logits.append(avg)
            predicted_labels.append(np.argmax(avg))
        video_names.append(base_name)
        # optional: get true label from one JSON (assuming same order)
        index = logits_v0["video_names"].index(base_name + "_view0")
        true_labels.append(logits_v1["true_labels"][index])
        
    #================================LAST REPORTS=============================
    rep_last_final = classification_report(true_labels, predicted_labels)
    rep_last_final_just012 = classification_report(true_labels, predicted_labels, labels=[l for l in const.LABELS_INCLUDED_IN_F1_CALCULATION if l in true_labels])
    confusion_last_final = confusion_matrix(true_labels, predicted_labels)
    log_results(rep_last_final, rep_last_final_just012, confusion_last_final, 'last_report_allfolds.txt', 'last_report_allfolds_just012updrs.txt', 'last_confusion_matrix_allfolds.png', out_path)
    cls_rep = classification_report(true_labels, predicted_labels, labels=[l for l in const.LABELS_INCLUDED_IN_F1_CALCULATION if l in true_labels], output_dict=True)
    wandb.log({
        'macro avg precision': cls_rep['macro avg']['precision'],
        'macro avg recall': cls_rep['macro avg']['recall'],
        'macro avg f1-score': cls_rep['macro avg']['f1-score'],
        'weighted avg precision': cls_rep['weighted avg']['precision'],
        'weighted avg recall': cls_rep['weighted avg']['recall'],
        'weighted avg f1-score': cls_rep['weighted avg']['f1-score'],
    })
    wandb.finish()