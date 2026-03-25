
import numpy as np


def get_array_block_sizes(
    shape: tuple[int], dtype: np.number | np.str_, max_array_size: int
) -> tuple[list[list[int]], list[list[int]]]:
    # Total size of array in bytes.
    array_size = int(np.prod(shape) * dtype.itemsize)
    # Calculate the minimum number of blocks needed (if the array was flat).
    num_blocks = int(np.ceil(array_size / max_array_size))

    # Check if we can split on the first axis.
    if num_blocks > shape[0]:
        assert len(shape) > 1
        # Recursively get block sizes on higher axes.
        starts, counts = get_array_block_sizes(shape[1:], dtype, max_array_size)
        # Repeat starts and counts from higher axes for each axis 0.
        starts = [[i] + s for i in range(shape[0]) for s in starts]
        counts = [[1] + c for i in range(shape[0]) for c in counts]

        return starts, counts

    # Count the number of axis elements (e.g., rows for a 2d-array) in each
    # block, and count the number of blocks that remain.
    num_elements_in_block, num_remainder = divmod(shape[0], num_blocks)

    # Get the number of extra blocks needed to fill in the remaining elements.
    num_extra_blocks = num_remainder // num_elements_in_block + int(
        num_remainder % num_elements_in_block > 0
    )
    # Count the number of elements in the last block.
    num_elements_in_last_block = num_remainder % num_elements_in_block
    # Increase the number of blocks to fit the remaining elements.
    num_blocks += num_extra_blocks

    # Verify that we still have more axis elements than blocks.
    assert num_blocks <= shape[0]

    # Set up the number of axis elements in each block.
    axis_counts = np.ones(num_blocks, dtype=int) * num_elements_in_block
    if num_elements_in_last_block > 0:
        assert num_elements_in_last_block < num_elements_in_block
        # Alter the last block with the remaining number of elements.
        axis_counts[-1] = num_elements_in_last_block

    # Create an array with starting indices for each block and a corresponding
    # array with the number of elements in each block.
    starts = np.zeros((num_blocks, len(shape)), dtype=int)
    counts = np.zeros_like(starts)

    # Sum up the number of element counts to get the starting index in each
    # block (starting at 0).
    starts[1:, 0] = np.cumsum(axis_counts[:-1])

    # The axis_counts already lists the number of elements in the first axis,
    # so we only add the length of each remaining axis as the counts for the
    # last axes.
    counts[:, 0] = axis_counts
    counts[:, 1:] = shape[1:]

    # Check that no block exceeds the maximum size.
    count_size = np.prod(counts, axis=1) * dtype.itemsize
    assert np.all(count_size - max_array_size <= 0)

    return starts.tolist(), counts.tolist()
