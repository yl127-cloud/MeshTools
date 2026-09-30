from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def make_frame():
    def factory(positions, *, images=None, box=(10, 10, 10, 0, 0, 0)):
        positions_array = np.asarray(positions, dtype=float)
        if images is None:
            images_array = np.zeros_like(positions_array, dtype=int)
        else:
            images_array = np.asarray(images, dtype=int)
        return SimpleNamespace(
            particles=SimpleNamespace(
                position=positions_array,
                image=images_array,
            ),
            configuration=SimpleNamespace(box=np.asarray(box, dtype=float)),
        )

    return factory

