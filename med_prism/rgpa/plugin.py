"""External plugin loaded AFTER the unmodified shared/private plugin."""
import json
import os
from pathlib import Path
import torch

from swift.trainers import TrainerFactory
from scripts.med_prism_real_5skill.reload_probe_plugin import MedPrismReloadProbeTrainer
from med_prism.rgpa.core import Protection
from med_prism.rgpa.state import write_json, save_resume, load_resume
from med_prism.transport.checkpoint import sha256_file


class RGPATrainer(MedPrismReloadProbeTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        c = self.sp_config
        if (c.orth_lambda != 0 or c.key_lambda != .1 or c.shared_drift_lambda != .01
                or c.key_loss_mode != 'rms_cosine_A'):
            raise ValueError('RGPA requires the audited current-best no-geo recipe')
        root = Path(os.environ['MED_PRISM_RGPA_ROOT'])
        histories = {str(i): json.loads((root / 'rgpa' / f'task_{i:02d}.json').read_text())
                     for i in range(1, c.current_task_id)}
        for i, manifest in enumerate(c.private_source_manifests, 1):
            if histories[str(i)]['source_private_sha256'] != sha256_file(manifest):
                raise ValueError('Salience does not belong to this historical private bank')
        resume = os.environ.get('MED_PRISM_RGPA_RESUME')
        saved = json.loads((Path(resume) / 'rgpa_state.json').read_text()) if resume else None
        self.protection = Protection(self.accelerator.unwrap_model(self.model), c.current_task_id,
                                    histories, slack=os.environ.get('MED_PRISM_RGPA_SLACK', '1') == '1',
                                    eps=c.orth_eps, saved=saved)
        if self.is_world_process_zero():
            write_json(self.sp_output / 'rgpa_initial_state.json', self.protection.state())
            def summary(values):
                t = torch.tensor(values, dtype=torch.float32)
                return dict(min=float(t.min()), median=float(t.median()), max=float(t.max())) if t.numel() else None
            taus = [v for _, values in self.protection.columns for v in values]
            write_json(self.sp_output / 'rgpa_scale_summary.json', dict(
                sigma=summary(list(self.protection.sigmas.values())), tau=summary(taus),
                zero_tau_fraction=sum(v == 0 for v in taus) / len(taus) if taus else None))

    def compute_loss(self, *args, **kwargs):
        with self.protection.use():
            return super().compute_loss(*args, **kwargs)

    def train(self, *args, **kwargs):
        resume = os.environ.get('MED_PRISM_RGPA_RESUME')
        if resume:
            if args:
                args = (resume, *args[1:])
            else:
                kwargs['resume_from_checkpoint'] = resume
        return super().train(*args, **kwargs)

    def _save_checkpoint(self, model, trial, *args, **kwargs):
        result = super()._save_checkpoint(model, trial, *args, **kwargs)
        if self.is_world_process_zero():
            directory = Path(self._get_output_dir(trial=trial)) / f'checkpoint-{self.state.global_step}'
            save_resume(self.accelerator.unwrap_model(self.model), self.protection, directory)
        return result

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        restored = load_resume(self.accelerator.unwrap_model(model or self.model), resume_from_checkpoint)
        if restored != self.protection.state():
            raise ValueError('Checkpoint RGPA state differs from task initialization')
        # HF Trainer restores optimizer/scheduler/RNG and data position separately.


_previous = TrainerFactory.get_trainer_cls.__func__


def _get(cls, args):
    if getattr(args, 'tuner_type', None) == 'med_prism_shared_private':
        return RGPATrainer
    return _previous(cls, args)


TrainerFactory.get_trainer_cls = classmethod(_get)
