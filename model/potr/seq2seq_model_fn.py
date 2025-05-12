###############################################################################
# Pose Transformers (POTR): Human Motion Prediction with Non-Autoregressive 
# Transformers
# 
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by 
# Angel Martinez <angel.martinez@idiap.ch>,
# 
# This file is part of 
# POTR: Human Motion Prediction with Non-Autoregressive Transformers
# 
# POTR is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
# 
# POTR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with POTR. If not, see <http://www.gnu.org/licenses/>.
###############################################################################

"""Implements a model function estimator for training, evaluation and predict.

Take and adapted from the code presented in [4]

[1] https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-249404882
[2] https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/motionGenerationError.m#L40-L54
[3] https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-247769197
[4] https://arxiv.org/pdf/1705.02445.pdf
"""

import sys
import numpy as np
import sys
import os
import time
from abc import abstractmethod
import tqdm
from copy import deepcopy

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# import wandb

thispath = os.path.dirname(os.path.abspath(__file__))                            
sys.path.insert(0, thispath+"/../")

import utils.WarmUpScheduler as warm_up_scheduler



_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# min threshold for mean average precision in metters
# Set to 10 cm
_MAP_TRESH = 0.10

class ModelFn(object):
  """Implements the model functionalities: training, evaliation and prediction."""

  def __init__(
      self,
      params,
      train_dataset_fn=None,
      test_dataset_fn=None,
      val_dataset_fn=None,
      pose_encoder_fn=None,
      fold=None):
    """Initialization of model function."""
    self._params = params
    self._train_dataset_fn = train_dataset_fn
    self._test_dataset_fn = test_dataset_fn
    self._val_dataset_fn = val_dataset_fn
    self._visualize = False
    thisname = self.__class__.__name__
    # self._norm_stats = train_dataset_fn.dataset._norm_stats
    self._norm_stats = None
    self.init_model(pose_encoder_fn)
    self._loss_fn = self.loss_mse
    self._model.to(_DEVICE)
    self._optimizer_fn = self.select_optimizer()
    self.select_lr_fn()
    self.finetune_init()
    self.fold = fold

    self._lr_db_curve = []

    lr_type = 'stepwise' if self._params['learning_rate_fn'] == 'beatles' \
        else 'epochwise'
    self._params['lr_schedule_type'] = lr_type

    self.evaluate_fn = self.evaluate_nturgbd

    self._time_range_eval = []

    m_params = filter(lambda p: p.requires_grad, self._model.parameters())
    nparams = sum([np.prod(p.size()) for p in m_params])        


  def finetune_init(self):
    if self._params['finetuning_ckpt'] is not None:
      print('[INFO] (finetune_model) Finetuning from:', 
          self._params['finetuning_ckpt'])

      # edits made here to exclude activity prediction head
      model_state_dict = torch.load(self._params['finetuning_ckpt'], map_location=_DEVICE)
      if 'gait' in self._params['dataset']: # exclude prediction head
        if not self._params['test_only']:
          del model_state_dict['_action_head.0.weight']
          del model_state_dict['_action_head.0.bias']
        self._model.load_state_dict(model_state_dict, strict=False)
      else:
        self._model.load_state_dict(model_state_dict)


  def select_lr_fn(self):
    """Calls the selection of learning rate function."""
    self._lr_scheduler = self.get_lr_fn()
    lr_fn = self._params['learning_rate_fn']
    if self._params['warmup_epochs'] > 0 and lr_fn != 'beatles':
      self._lr_scheduler = warm_up_scheduler.GradualWarmupScheduler(
          self._optimizer_fn, multiplier=1, 
          total_epoch=self._params['warmup_epochs'], 
          after_scheduler=self._lr_scheduler
      )

  def get_lr_fn(self):
    """Creates the function to be used to generate the learning rate."""
    if self._params['learning_rate_fn'] == 'step':
      return torch.optim.lr_scheduler.StepLR(
        self._optimizer_fn, step_size=self._params['lr_step_size'], gamma=0.1
      )
    elif self._params['learning_rate_fn'] == 'exponential':
      return torch.optim.lr_scheduler.ExponentialLR(
        self._optimizer_fn, gamma=0.95
      )
    elif self._params['learning_rate_fn'] == 'linear':
      # sets learning rate by multipliying initial learning rate times a function
      lr0, T = self._params['learning_rate'], self._params['max_epochs']
      lrT = lr0*0.5
      m = (lrT - 1) / T
      lambda_fn =  lambda epoch: m*epoch + 1.0
      return torch.optim.lr_scheduler.LambdaLR(
        self._optimizer_fn, lr_lambda=lambda_fn
      )
    elif self._params['learning_rate_fn'] == 'beatles':
      # D^(-0.5)*min(i^(-0.5), i*warmup_steps^(-1.5))
      D = float(self._params['model_dim'])
      warmup = self._params['warmup_epochs']
      lambda_fn = lambda e: (D**(-0.5))*min((e+1.0)**(-0.5), (e+1.0)*warmup**(-1.5))
      return torch.optim.lr_scheduler.LambdaLR(
        self._optimizer_fn, lr_lambda=lambda_fn
      )
    else:
      raise ValueError('Unknown learning rate function: {}'.format(
          self._params['learning_rate_fn']))

  @abstractmethod
  def init_model(self, pose_encoder_fn):
    pass

  @abstractmethod
  def select_optimizer(self):
    pass

  def loss_mse(self, decoder_pred, decoder_gt):
    """Computes the L2 loss between predictions and ground truth."""
    step_loss = (decoder_pred - decoder_gt)**2
    step_loss = step_loss.mean()

    return step_loss

  @abstractmethod
  def compute_loss(self, inputs=None, target=None, preds=None, class_logits=None, class_gt=None):
    return self._loss_fn(preds, target, class_logits, class_gt)

  def print_logs(self, step_loss, current_step, pose_loss, activity_loss, selection_loss):
    selection_logs = ''
    if self._params['query_selection']:
      selection_logs = 'selection loss {:.4f}'.format(selection_loss)
    if self._params['predict_activity']:
      print("[INFO] global {:06d}; step {:04d}; pose_loss {:4f} - class_loss {:4f}; step_loss: {:.4f}; lr: {:.2e} {:s}".\
          format(self._global_step, current_step, pose_loss, activity_loss, 
                step_loss, self._params['learning_rate'], selection_logs) 
      )
    else:
      print("[INFO] global {3:06d}; step {0:04d}; step_loss: {1:.4f}; lr: {2:.2e} {4:s}".\
          format(current_step, step_loss, self._params['learning_rate'], 
              self._global_step, selection_logs)
    )


  def train_one_epoch(self, epoch):
    """Trains for a number of steps before evaluation."""
    epoch_loss = 0
    act_loss = 0
    sel_loss = 0
    N = len(self._train_dataset_fn)
    for current_step, sample in enumerate(self._train_dataset_fn):
      self._optimizer_fn.zero_grad()
      for k in sample.keys():
        if k == 'actions' or k == 'decoder_outputs_euler' or k=='labels_str':
          continue
        sample[k] = sample[k].to(_DEVICE)

      decoder_pred = self._model(sample['encoder_inputs'])

      # self._writer.add_graph(self._model, (sample['encoder_inputs'], sample['decoder_inputs']))

      pred_class, gt_class = None, None
      if self._params['predict_activity']:
        gt_class = sample['labels']  # one label for the sequence
        pred_class = decoder_pred[1]

      activity_loss = self.compute_loss(
          inputs=sample['encoder_inputs'],
          class_logits=pred_class,
          class_gt=gt_class
      )

      step_loss = activity_loss
      if self._params['predict_activity']:
        if self._params['task'] == 'pretext':
          step_loss += self._params['activity_weight']*activity_loss
        else:
          if self._params['downstream_strategy'] == 'both':
            step_loss += self._params['activity_weight']*activity_loss
          elif self._params['downstream_strategy'] == 'class':
            step_loss = activity_loss
          elif self._params['downstream_strategy'] == 'both_then_class':
            if epoch >= 50:
              step_loss = activity_loss
            else:
              step_loss += self._params['activity_weight']*activity_loss
              
        act_loss += activity_loss.item()
      epoch_loss += step_loss.item()

      step_loss.backward()
      if self._params['max_gradient_norm'] is not None:
        torch.nn.utils.clip_grad_norm_(
          self._model.parameters(), self._params['max_gradient_norm'])
      self._optimizer_fn.step()

      if current_step % 10 == 0:
        step_loss = step_loss.cpu().data.numpy()
        # self.print_logs(step_loss, current_step, pose_loss, activity_loss, 
        #     selection_loss)

      self.update_learning_rate(self._global_step, mode='stepwise')
      self._global_step += 1

    if self._params['query_selection']:
      self._scalars['train_selectioin_loss'] = sel_loss/N

    if self._params['predict_activity']:
      return epoch_loss/N, act_loss/N

    return epoch_loss/N

  def early_stopping(self, patience, best_loss, current_loss, delta, wait):
    """
    Performs early stopping based on validation loss.

    Args:
        patience (int): Number of epochs to wait for improvement before stopping.
        best_loss (float): Current best validation loss.
        current_loss (float): Current validation loss.
        delta (float): Minimum change in loss required to be considered as improvement.

    Returns:
        Tuple: A tuple with two elements, where the first element is a boolean indicating whether to early stop or not,
        and the second element is the updated best loss.
    """
    if current_loss < best_loss - delta:
      # If current_loss is better than best_loss - delta, update best_loss to current_loss
      best_loss = deepcopy(current_loss)
      wait = 0
    else:
      # If current_loss is not better than best_loss - delta, increment wait counter
      wait += 1

    # If wait counter exceeds patience, return True to early stop, else return False
    if wait > patience:
      return True, best_loss, wait
    else:
      return False, best_loss, wait

  def train(self):
    """Main training loop."""
    self._params['learning_rate'] = self._lr_scheduler.get_lr()[0]
    self._global_step = 1
    thisname = self.__class__.__name__


    # wandb.init(name='training', project='GaitForeMer')
    # best_class_loss = 1000

    # for name, param in self._model.named_parameters():
    #   print(name)
    #   print(param.requires_grad)
    
    for e in range(self._params['max_epochs']):
      self._scalars = {}
      self._model.train()
      start_time = time.time()
      epoch_loss = self.train_one_epoch(e)

      act_log = ''
      if self._params['predict_activity']:
        act_loss = epoch_loss[1]
        epoch_loss = epoch_loss[0]
        act_log = '; activity_loss: {}'.format(act_loss)
        self._scalars['act_loss_train'] = act_loss

      self._scalars['epoch_loss'] = epoch_loss
      print("epoch {0:04d}; epoch_loss: {1:.4f}".format(e, epoch_loss)+act_log)
      self.flush_extras(e, 'train')

      _time = time.time() - start_time


      self._model.eval()
      test_loss = self.evaluate_fn(e, _time, 'test')
      # if test_loss[1].item() <= best_class_loss:
      #   best_class_loss = test_loss[1].item()
      #   best_class_pred = test_loss[3]
      val_loss = self.evaluate_fn(e, _time, 'val')
      if e == 0:
        best_loss = deepcopy(val_loss[1])
        wait = 0
      else:
        early_stop, best_loss, wait = self.early_stopping(self._params['patience'], best_loss, val_loss[1], self._params['delta'], wait)
        if early_stop:
          print("Early stopping triggered at epoch", e)
          model_path = os.path.join(
            self._params['model_prefix'], 'models', 'ckpt_epoch_%04d_fold_%02d_best.pt' % (e, self.fold))
          torch.save(self._model.state_dict(), model_path)
          break

      act_log = ''
      if self._params['predict_activity']:
        self._scalars['act_loss_test'] = test_loss[1]
        self._scalars['test_accuracy'] = test_loss[2]
        act_log = '; act_test_loss {}; test_accuracy {}'.format(test_loss[1], test_loss[2])
        test_activity_loss = test_loss[1]
        test_accuracy = test_loss[2]
        # test_loss = test_loss[0]
        self._scalars['act_loss_eval'] = val_loss[1]
        self._scalars['val_accuracy'] = val_loss[2]
        act_log += '; act_val_loss {}; val_accuracy {}'.format(val_loss[1], val_loss[2])
        eval_activity_loss = val_loss[1]
        eval_accuracy = val_loss[2]

      self._scalars['test_loss'] = test_loss[0]
      self._scalars['val_loss'] = val_loss[0]
      print("[INFO] ({}) Epoch {:04d}; test_loss: {:.4f}; val_loss: {:.4f}; lr: {:.2e}".format(
          thisname, e, test_loss[0], val_loss[0], self._params['learning_rate'])+act_log)


      self.write_summary(e)

      # wandb_logs = {"train loss": epoch_loss, "train activity loss": act_loss, "eval loss": test_loss, "eval activity loss":  eval_activity_loss, "eval accuracy": eval_accuracy}
      # wandb.log(wandb_logs)


      model_path = os.path.join(
          self._params['model_prefix'], 'models', 'ckpt_epoch_%04d_fold_%02d.pt' % (e, self.fold))
      if (e+1)%100 == 0:
        torch.save(self._model.state_dict(), model_path)

      self.update_learning_rate(e, mode='epochwise')
      self.flush_extras(e, 'eval')

    # return predictions and real ones
    predictions = test_loss[3]  #best_class_pred
    gt = test_loss[4]
    pred_probs = test_loss[5]


    return predictions, gt, pred_probs

    # save the last one
    # model_path = os.path.join(
    #     self._params['model_prefix'], 'models', 'ckpt_epoch_%04d.pt'%e)
    # torch.save(self._model.state_dict(). model_path)
    # self.flush_curves()

  def write_summary(self, epoch):
    # for action_, ms_errors_ in ms_eval_loss.items():
    self._writer.add_scalars(
       'fold{}/loss/recon_loss'.format(self.fold),
        {'train':self._scalars['epoch_loss'], 'eval': self._scalars['val_loss'], 'test': self._scalars['test_loss']},
        epoch
    )

    # write scalars for H36M dataset prediction style
    action_ = self._train_dataset_fn.dataset._monitor_action
    if 'ms_eval_loss' in self._scalars.keys():
      range_len = len(self._scalars['ms_eval_loss'][action_])
      # range_len = len(self._ms_range)
      ms_dict = {str(self._ms_range[i]): self._scalars['ms_eval_loss'][action_][i] 
                 for i in range(range_len)}
      ms_e = np.concatenate([np.array(v).reshape(1,range_len) 
                            for k,v in self._scalars['ms_eval_loss'].items()], axis=0)
      self._writer.add_scalars('ms_loss/eval-'+action_, ms_dict, epoch)

      ms_e = np.mean(ms_e, axis=0)  # (n_actions)
      self._time_range_eval.append(np.expand_dims(ms_e, axis=0)) # (1, n_actions)
      all_ms = {str(self._ms_range[i]): ms_e[i] for i in range(len(ms_e))}
      self._writer.add_scalars('ms_loss/eval-all', all_ms, epoch)

      self._writer.add_scalar('MSRE/msre_eval', self._scalars['msre'], epoch)
      self._writer.add_scalars('time_range/eval', 
          {'short-term':np.mean(ms_e[:4]), 'long-term':np.mean(ms_e)}, epoch)

    if self._params['predict_activity']:
      self._writer.add_scalars(
          'fold{}/loss/class_loss'.format(self.fold),
          {'train': self._scalars['act_loss_train'], 'eval': self._scalars['act_loss_eval']}, 
          epoch
      )
      self._writer.add_scalar('fold{}/class/accuracy'.format(self.fold), self._scalars['test_accuracy'], epoch)
      self._writer.add_scalar('fold{}/class/accuracy'.format(self.fold), self._scalars['val_accuracy'], epoch)

    if self._params['query_selection']:
      self._writer.add_scalars(
          'fold{}/selection/query_selection'.format(self.fold),
          {'eval': self._scalars['eval_selection_loss'], 
           'train': self._scalars['train_selectioin_loss']},
          epoch
      )

    if 'mAP' in self._scalars.keys():
      self._writer.add_scalar('mAP/mAP', self._scalars['mAP'], epoch)

    if 'MPJPE' in self._scalars.keys():
      self._writer.add_scalar('MPJPE/MPJPE', self._scalars['MPJPE'], epoch)

  def flush_curves(self):
    path_ = os.path.join(self._params['model_prefix'], 'loss_info')
    os.makedirs(path_, exist_ok=True)
    path_ = os.path.join(path_, 'eval_time_range.npy')
    np.save(path_, np.concatenate(self._time_range_eval, axis=0))
    path_ = os.path.join(path_, 'lr_schedule.npy')
    np.save(path_, np.array(self._lr_db_curve))

  def update_learning_rate(self, epoch_step, mode='stepwise'):
    """Update learning rate handler updating only when the mode matches."""
    if self._params['lr_schedule_type'] == mode:
      self._lr_scheduler.step(epoch_step)
      self._writer.add_scalar(
          'learning_rate/lr', self._params['learning_rate'], epoch_step)
      self._lr_db_curve.append([self._params['learning_rate'], epoch_step])
      self._params['learning_rate'] = self._lr_scheduler.get_lr()[0]

  @abstractmethod
  def flush_extras(self, epoch, phase):
    pass

  def compute_class_accurracy_sequence(self, class_logits, class_gt):
    # softmax on last dimension and get max on last dimension
    class_pred = torch.argmax(class_logits.softmax(-1), -1)
    accuracy = (class_pred == class_gt).float().sum()
    accuracy = accuracy / class_logits.size()[0]
    return accuracy.item()


  @torch.no_grad()
  def evaluate_nturgbd(self, current_step, dummy_entry=None, mode='val'):
    eval_loss = 0.0
    mAP_all = 0.0
    class_loss = 0.0
    mean_accuracy = 0.0

    if mode == 'test':
      Data = deepcopy(self._test_dataset_fn)
    if mode == 'val':
      Data = deepcopy(self._val_dataset_fn)

    N = len(Data)
    gt_class_ = []
    pred_class_ = []

    num_joints = self._params['pose_dim'] // 3
    TP = np.zeros((num_joints,))
    FN = np.zeros((num_joints,))
    MPJPE = np.zeros((num_joints,))

    for (i, sample) in tqdm.tqdm(enumerate(Data)):
      for k in sample.keys():
        if k=='labels_str':
          continue
        sample[k] = sample[k].to(_DEVICE)

      decoder_pred = self._model(
          sample['encoder_inputs'], sample['decoder_inputs'])

      pred_class, gt_class = None, None
      if self._params['predict_activity']:
        gt_class = sample['labels']  # one label for the sequence
        pred_class = decoder_pred[1]
        decoder_pred = decoder_pred[0]
        gt_class_.append(gt_class.item())
        pred_class_.append(pred_class[-1].cpu().numpy())

      pose_loss, activity_loss = self.compute_loss(
          inputs=sample['encoder_inputs'],
          target=sample['decoder_outputs'],
          preds=decoder_pred,
          class_logits=pred_class,
          class_gt=gt_class
      )

      # Can save predicted outputs for visualization here
      # if i == 2:
      #   predicted_pose = decoder_pred[-1].squeeze().reshape(20, 17, 3).cpu().numpy()
      #   input_pose = sample['encoder_inputs'].squeeze().reshape(39, 17, 3).cpu().numpy()
      #   gt_pose = sample['decoder_outputs'].squeeze().reshape(20, 17, 3).cpu().numpy()
      #   np.save('output_poses/v37_pred.npy', predicted_pose)
      #   np.save('output_poses/v37_gt.npy', gt_pose)
      #   np.save('output_poses/v37_input.npy', input_pose)

      #   # break


      eval_loss+= pose_loss
      class_loss += activity_loss

    eval_loss /= N

    if self._params['predict_activity']:
      class_loss /= N
      pred_class_ = torch.squeeze(torch.from_numpy(np.stack(pred_class_)))
      gt_class_ = torch.from_numpy(np.array(gt_class_))
      # print(pred_class_.size(), gt_class_.size())
      accuracy = self.compute_class_accurracy_sequence(pred_class_, gt_class_)

      return (eval_loss, class_loss, accuracy, torch.argmax(pred_class_.softmax(-1), -1), gt_class_, pred_class_.softmax(-1))
      # return (eval_loss, class_loss, accuracy, torch.argmax(pred_class_.softmax(-1), -1), gt_class_)
      # return (eval_loss, class_loss, accuracy)

    return eval_loss


  def test(self):
    """Test function"""
    # wandb.init(name='training', project='GaitForeMer')

    self._model.eval()
    eval_loss = self.evaluate_fn(-1, -1)
    self._scalars = {}

    act_log = ''
    if self._params['predict_activity']:
      self._scalars['act_loss_eval'] = eval_loss[1]
      self._scalars['accuracy'] = eval_loss[2]
      act_log = '; act_eval_loss {}; accuracy {}'.format(eval_loss[1], eval_loss[2])
      eval_activity_loss = eval_loss[1]
      eval_accuracy = eval_loss[2]
      # eval_loss = eval_loss[0]

    self._scalars['eval_loss'] = eval_loss[0]

    # return predictions and real ones
    predictions = eval_loss[3]
    gt = eval_loss[4]
    pred_probs = eval_loss[5]

    return predictions, gt, pred_probs



