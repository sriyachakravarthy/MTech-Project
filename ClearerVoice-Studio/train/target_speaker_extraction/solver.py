import time
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
 
 
 
from losses.loss_function import loss_wrapper
from losses.metrics import SDR, cal_SISNR
from pystoi import stoi
from pesq import pesq
 
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import numpy as np
 
from losses.loss_function import loss_wrapper
from losses.metrics import SDR, cal_SISNR
from pystoi import stoi
from pesq import pesq
 
class Solver(object):
    # how many epochs to linearly ramp clip‐weight 0→1
    DEFAULT_CLIP_RAMP_EPOCHS = 10
 
    def __init__(self, args, model, optimizer, train_data, validation_data, test_data):
        # store handles
        self.args            = args
        self.train_data      = train_data
        self.validation_data = validation_data
        self.test_data       = test_data
 
        # primary audio loss and CLIP‐style EEG→speech projection
        self.loss     = loss_wrapper(args.loss_type)
        self.eeg_proj = nn.Linear(64, 256).to(args.device)
        self.audio_proj = nn.Linear(256, 256).to(args.device)

 
        # how many epochs to ramp in clip loss
        self.clip_ramp_epochs = getattr(args, 'clip_ramp_epochs',
                                        self.DEFAULT_CLIP_RAMP_EPOCHS)
 
        # only print/log from rank0 or non-distributed runs
        self.print = (not getattr(args, 'distributed', False)) or \
                     (getattr(args, 'distributed', False) and args.local_rank == 0)
        if self.print and not getattr(args, 'evaluate_only', False):
            self.writer = SummaryWriter(f"{args.checkpoint_dir}/tensorboard/")
 
        # wrap model for DDP if needed
        self.model     = model.to(args.device)
        self.optimizer = optimizer
        if getattr(args, 'distributed', False):
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model,
                             device_ids=[args.local_rank],
                             find_unused_parameters=True)
 
        # initialize or load checkpoint state
        if not getattr(args, 'evaluate_only', False):
            self._init_training_state()
 
 
    def _init_training_state(self):
        """Set up counters, optionally resume from checkpoint."""
        self.halving       = False
        self.step_num      = 1
        self.best_val_loss = float('inf')   # tracked on val_audio
        self.val_no_impv   = 0
        self.start_epoch   = 1
        self.epoch         = 0
 
        if getattr(self.args, 'train_from_last_checkpoint', False):
            self._load_model(f"{self.args.checkpoint_dir}/last_checkpoint.pt",
                              load_training_stat=True)
        elif getattr(self.args, 'init_from', 'None') != 'None':
            self._load_model(f"{self.args.init_from}/last_best_checkpbest_oint.pt")
            if self.print:
                print(f"Init from {self.args.init_from}, starting fresh training")
        else:
            if self.print:
                print("Start new training from scratch")
 
        # save a warm‐start checkpoint
        self._save_model(f"{self.args.checkpoint_dir}/last_checkpoint.pt")
 
    def _clip_loss(self, eeg_embed, speech_embed):
        """
        Symmetric InfoNCE between EEG and speech embeds.
        """
        # [B, C, T] → project → [B, C, T] → mean T → [B, C]
        eeg = eeg_embed.permute(0,2,1)
        eeg = self.eeg_proj(eeg).permute(0,2,1).mean(dim=2)
        eeg = F.normalize(eeg, dim=1)
 
        # Speech: [B, C, T] -> mean -> [B, C] -> project -> [B, 512]
        sp = speech_embed.mean(dim=2)           # [B, C]
        sp = self.audio_proj(sp)                # [B, 512]
        sp = F.normalize(sp, dim=1)
 
        logits = torch.matmul(eeg, sp.T) / 0.07
        labels = torch.arange(logits.size(0), device=logits.device)
 
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return 0.5*(loss_i2t + loss_t2i)
     
    def train(self):
        """Full train‐val‐test loop over epochs, with early stopping on audio loss."""
        for self.epoch in range(self.start_epoch, self.args.max_epoch + 1):
            # for DDP: make sampler shuffle differently each epoch
            if getattr(self.args, 'distributed', False):
                self.args.train_sampler.set_epoch(self.epoch)
 
            # ----- TRAIN -----
            self.model.train()
            t0 = time.time()
            tr_total, tr_audio, tr_clip = self._run_one_epoch(self.train_data,
                                                              state='train')
            if getattr(self.args, 'distributed', False):
                tr_audio = self._reduce_tensor(tr_audio)
            if self.print:
                print(f"[Epoch {self.epoch}] TRAIN in {time.time()-t0:.1f}s  "
                      f"total={tr_total:.4f} audio={tr_audio:.4f} clip={tr_clip:.4f}")
                self.writer.add_scalar('Train/Audio_Loss', tr_audio, self.epoch)
                self.writer.add_scalar('Train/Clip_Loss',  tr_clip,  self.epoch)
 
            # ----- VALID -----
            self.model.eval()
            with torch.no_grad():
                t0 = time.time()
                val_total, val_audio, val_clip = self._run_one_epoch(self.validation_data,
                                                                    state='val')
                if getattr(self.args, 'distributed', False):
                    val_audio = self._reduce_tensor(val_audio)
            if self.print:
                print(f"[Epoch {self.epoch}] VALID in {time.time()-t0:.1f}s  "
                      f"audio={val_audio:.4f} total={val_total:.4f}")
                self.writer.add_scalar('Valid/Audio_Loss', val_audio, self.epoch)
 
            # ----- TEST -----
            with torch.no_grad():
                t0 = time.time()
                test_total, test_audio, test_clip = self._run_one_epoch(self.test_data,
                                                                        state='test')
                if getattr(self.args, 'distributed', False):
                    test_audio = self._reduce_tensor(test_audio)
            if self.print:
                print(f"[Epoch {self.epoch}]  TEST in {time.time()-t0:.1f}s  "
                      f"audio={test_audio:.4f}")
                self.writer.add_scalar('Test/Audio_Loss', test_audio, self.epoch)
 
            # --- Early stopping & LR halving on val_audio ---
            improved = val_audio < self.best_val_loss
            if improved:
                self.best_val_loss = val_audio
                self.val_no_impv   = 0
            else:
                self.val_no_impv += 1
                if self.val_no_impv == 5:
                    self.halving = True
                elif self.val_no_impv >= 20:
                    if self.print:
                        print("No improvement for210 epochs → early stopping")
                    break
 
            # --- Halve LR if needed ---
            if self.halving:
                self.halving = False
                # reload best checkpoint's optimizer state
                self._load_model(f"{self.args.checkpoint_dir}/last_best_checkpoint.pt",
                                  load_optimizer=True)
                sd = self.optimizer.state_dict()
                sd['param_groups'][0]['lr'] *= 0.5
                self.optimizer.load_state_dict(sd)
                if self.print:
                    print(f"Halved LR → {sd['param_groups'][0]['lr']:.6f}")
 
            # --- Save checkpoints ---
            if self.print:
                self._save_model(f"{self.args.checkpoint_dir}/last_checkpoint.pt")
                if improved:
                    self._save_model(f"{self.args.checkpoint_dir}/last_best_checkpoint.pt")
                    print("Saved new best checkpoint")
 
 
    def _run_one_epoch(self, data_loader, state='train'):
        """
        Runs one pass through data_loader.
        Returns: (avg_total_loss, avg_audio_loss, avg_clip_loss)
        """
        total_loss       = 0.0
        total_audio_loss = 0.0
        total_clip_loss  = 0.0
        self.accu_count  = 0
        self.optimizer.zero_grad()
 
        # linear ramp of clip weight from 0→1 over clip_ramp_epochs
        clip_w  = 0 #min(self.epoch / float(self.clip_ramp_epochs), 1.0)
        audio_w = 1.0 #- clip_w
 
        for i, (a_mix, a_tgt, ref_tgt) in enumerate(data_loader):
            iter_start = time.time()
            a_mix, a_tgt = a_mix.to(self.args.device), a_tgt.to(self.args.device)

            # forward (model must return audio_est, speech_embed, eeg_embed)
            a_tgt_est, speech_embed, eeg_embed = self.model(a_mix, ref_tgt)
            speech_embed, eeg_embed = speech_embed.to(self.args.device), eeg_embed.to(self.args.device)
 
            # compute losses
            clip_loss  = self._clip_loss(eeg_embed, speech_embed)
            audio_loss = self.loss(a_tgt, a_tgt_est)
            loss       = audio_w * audio_loss + clip_w * clip_loss
 
            if state == 'train':
                # gradient accumulation
                if getattr(self.args, 'accu_grad', False):
                    self.accu_count += 1
                    (loss / (self.args.effec_batch_size / self.args.batch_size)).backward()
                    if self.accu_count >= (self.args.effec_batch_size / self.args.batch_size):
                        if getattr(self.args, 'lr_warmup', False):
                            self._adjust_lr_warmup()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       self.args.clip_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.accu_count = 0
                else:
                    loss.backward()
                    if getattr(self.args, 'lr_warmup', False):
                        self._adjust_lr_warmup()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.args.clip_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
 
            # accumulate scalars
            total_loss       += loss.item()
            total_audio_loss += audio_loss.item()
            total_clip_loss  += clip_loss.item()
            n = i + 1
 
            iter_time = time.time() - iter_start
            if self.print and i % 100 == 0:
                print(f"[Iter {i:4d}] time={iter_time:.3f}s  avg_total={total_loss/n:.4f}, "
                    f"avg_audio={total_audio_loss/n:.4f}, avg_clip={total_clip_loss/n:.4f}  "
                    f"(audio_w={audio_w:.2f}, clip_w={clip_w:.2f})")

 
        return total_loss/n, total_audio_loss/n, total_clip_loss/n
 
 
 
 
    def _reduce_tensor(self, x):
        """
        All-reduce a scalar tensor in DDP; pass floats unchanged.
        """
        if not isinstance(x, torch.Tensor) or not getattr(self.args, 'distributed', False):
            return x
        rt = x.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.args.world_size
        return rt
 
 
    def _load_model(self, path, load_optimizer=False, load_training_stat=False):
        chk = torch.load(path, map_location='cpu')
        pretrained = chk['model']
        cur = self.model.state_dict()
        for k in cur:
            if k in pretrained and cur[k].shape == pretrained[k].shape:
                cur[k] = pretrained[k]
            elif f"module.{k}" in pretrained and cur[k].shape == pretrained[f"module.{k}"].shape:
                cur[k] = pretrained[f"module.{k}"]
        self.model.load_state_dict(cur)
 
        if load_optimizer:
            self.optimizer.load_state_dict(chk['optimizer'])
 
        if load_training_stat:
            self.step_num      = chk['step_num']
            self.best_val_loss = chk['best_val_loss']
            self.val_no_impv   = chk['val_no_impv']
            self.start_epoch   = chk['epoch']
            self.epoch         = self.start_epoch - 1
            if self.print:
                print(f"Resuming from epoch {self.start_epoch}")
 
 
    def _save_model(self, path):
        if not self.print:
            return
        chk = {
            'model':         self.model.state_dict(),
            'optimizer':     self.optimizer.state_dict(),
            'epoch':         self.epoch + 1,
            'step_num':      self.step_num,
            'best_val_loss': self.best_val_loss,
            'val_no_impv':   self.val_no_impv
        }
        torch.save(chk, path)
 
 
    def _adjust_lr_warmup(self):
        """
        Inverse‐sqrt warmup schedule.
        """
        warmup_steps = getattr(self, 'warmup_steps', 15000)
        if self.step_num < warmup_steps:
            lr = self.args.init_learning_rate / 0.001 \
                 * (64 ** -0.5) * self.step_num * (warmup_steps ** -1.5)
            self.step_num += 1
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
 
 
    def _print_lr(self):
        if self.print:
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            print(f"Current LR: {lr:.6f}")
 
 
 
    def evaluate(self, data_loader):
        avg_sisnri = 0
        avg_sdri = 0
        avg_pesqi = 0
        avg_stoii = 0
 
        self._load_model(f'{self.args.checkpoint_dir}/last_best_checkpoint.pt')
        self.model.eval()
        with torch.no_grad():
            for i, (a_mix, a_tgt, ref_tgt) in enumerate(data_loader):
                a_mix = a_mix.to(self.args.device)
                a_tgt = a_tgt.to(self.args.device)
 
                a_tgt_est,_,_ = self.model(a_mix, ref_tgt)
                #a_tgt_est = self.model(a_mix, ref_tgt)
 
                sisnri = cal_SISNR(a_tgt, a_tgt_est) - cal_SISNR(a_tgt, a_mix)
                avg_sisnri += sisnri
                # print(sisnri)
                a_tgt_est = a_tgt_est.squeeze().cpu().numpy()
                a_tgt = a_tgt.squeeze().cpu().numpy()
                a_mix = a_mix.squeeze().cpu().numpy()
 
                sdri = SDR(a_tgt, a_tgt_est) - SDR(a_tgt, a_mix)
                avg_sdri += sdri
 
                a_tgt_est = a_tgt_est/np.max(np.abs(a_tgt_est))
                pesqi =  (pesq(self.args.audio_sr, a_tgt, a_tgt_est,'nb') - pesq(self.args.audio_sr, a_tgt, a_mix,'nb'))
                avg_pesqi += pesqi
 
                stoii = (stoi(a_tgt, a_tgt_est, self.args.audio_sr, extended=False) - stoi(a_tgt, a_mix, self.args.audio_sr, extended=False))
                avg_stoii += stoii
 
 
        avg_sisnri = avg_sisnri / (i+1)
        avg_sdri = avg_sdri / (i+1)
        avg_pesqi = avg_pesqi / (i+1)
        avg_stoii = avg_stoii / (i+1)
 
 
        print(f'Avg SISNR:i {avg_sisnri}')
        print(f'Avg SNRi: {avg_sdri}')
        print(f'Avg PESQi: {avg_pesqi}')
        print(f'Avg STOIi: {avg_stoii}')
 