import numpy as np
from numpy.random import default_rng

rng = default_rng()


def rs(training_data, training_labels, img_per_group):
    # Randomly select from groups
    chosen_images = []
    for i in range(min(training_labels), max(training_labels) + 1):
        current_group = training_data[training_labels == i]
        if len(current_group) > img_per_group:
            chosen_images.extend(
                current_group[
                    rng.choice(
                        len(current_group), img_per_group, replace=False
                    )
                ]
            )
        else:
            chosen_images.extend(current_group)
    return chosen_images
