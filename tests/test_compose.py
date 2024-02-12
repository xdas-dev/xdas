import numpy as np

import xdas
import xdas.signal as xp


class TestCompose:

    sequence = xdas.Sequence(
        xdas.Atom(xp.taper, dim="time"),
        xdas.Atom(xp.taper, dim="distance"),
        xdas.Atom(np.abs),
        xdas.Atom(np.square),
    )

    def test_sequence_selection(self):
        for i, key in enumerate(self.sequence.keys()):
            assert self.sequence[i] == self.sequence[key]

    def test_sequence_order(self):

        sequence = self.sequence.copy()

        old_keys = np.array(list(sequence.keys()))

        sequence[0].move_up()
        new_keys = np.array(list(sequence.keys()))
        assert all(old_keys == new_keys)

        # sequence[-1].move_down()
        # new_keys = np.array(list(sequence.keys()))
        # assert all(old_keys == new_keys)

        sequence[1].move_up()
        new_keys = np.array(list(sequence.keys()))
        assert all(old_keys[[1, 0, 2, 3]] == new_keys)

        sequence[0].move_down()
        new_keys = np.array(list(sequence.keys()))
        assert all(old_keys == new_keys)



