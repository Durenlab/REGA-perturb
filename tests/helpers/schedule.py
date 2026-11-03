"""Short training schedules for tests (the package schedule is fixed)."""

from rega_perturb.model.train import _TrainingSchedule


def short_schedule(**kw):
    """Phase 1 = 5 epochs, phase 2 = 2 epochs (no convergence stop) unless overridden."""
    return _TrainingSchedule(**{**dict(phase1_epochs=5, phase2_min_epochs=2,
                                       phase2_max_epochs=2), **kw})
