"""Point-count weighting without retaining multiple forward graphs."""
from itertools import islice

import numpy as np


def coordinate_accumulation_batches(loader, accumulation_steps, extra_pde_points=0):
    """Yield CPU batches, separate data/PDE weights, and optimizer-step boundary.

    Groups end at the model-batch boundary. A short final group is normalized
    by its actual size and flushed, never carried into validation/another epoch.
    Extra PDE points are evaluated once per microbatch, as in model.loss.
    """
    if accumulation_steps < 1 or extra_pde_points < 0:
        raise ValueError('Invalid accumulation steps or extra PDE point count')
    iterator = iter(loader)
    while True:
        group = list(islice(iterator, accumulation_steps))
        if not group:
            return
        counts = [len(batch[0]) for batch in group]
        if min(counts) <= 0:
            raise ValueError('Empty coordinate microbatch')
        data_total = sum(counts)
        pde_total = data_total + len(group) * extra_pde_points
        for index, (batch, count) in enumerate(zip(group, counts)):
            yield (batch, count / data_total,
                   (count + extra_pde_points) / pde_total,
                   index == len(group) - 1)


def point_mean(values, counts):
    """Mean of microbatch means, weighted by model-coordinate pair counts."""
    if len(values) != len(counts) or not values or min(counts) <= 0:
        raise ValueError('Loss values require matching positive point counts')
    return float(np.average(values, weights=counts))


def weighted_microbatch_loss(loss, pde_loss, pde_coefficient, data_weight, pde_weight):
    # Auxiliary terms retain data-point weighting. PDE may include extra y_ran.
    return loss * data_weight + pde_coefficient * pde_loss * (pde_weight - data_weight)
