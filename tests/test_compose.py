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
        """Test selection by index vs. selection by key"""
        for i, key in enumerate(self.sequence.keys()):
            assert self.sequence[i] == self.sequence[key]

    def test_sequence_order(self):
        """Test Sequence order manipulations"""

        sequence = self.sequence.copy()
        old_keys = np.array(list(sequence.keys()))

        sequence[0].move_up()
        new_keys = np.array(list(sequence.keys()))
        assert all(old_keys == new_keys)

        sequence[-1].move_down()
        new_keys = np.array(list(sequence.keys()))
        assert all(old_keys == new_keys)

        sequence[1].move_up()
        new_keys = np.array(list(sequence.keys()))
        assert all(old_keys[[1, 0, 2, 3]] == new_keys)

        sequence[0].move_down()
        new_keys = np.array(list(sequence.keys()))
        assert all(old_keys == new_keys)

    def test_sequence_delete(self):
        """Test Atom deletion from Sequence"""

        sequence = self.sequence.copy()
        old_keys = list(sequence.keys())

        del sequence[2]
        new_keys = list(sequence.keys())
        old_keys.pop(2)

        assert old_keys == new_keys

    def test_sequence_insert(self):
        """Test Atom insertion into Sequence"""

        sequence1 = self.sequence.copy()

        # Test insertion via __setitem__
        sequence2 = xdas.Sequence()
        for key, atom in sequence1.items():
            sequence2[key] = atom

        assert sequence1 == sequence2

        # TODO: test insertion via insert_before/after