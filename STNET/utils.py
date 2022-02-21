def get_dilation(level, factor=2):
    return factor**level


def get_receptive_field_size(kernel_size, num_levels, num_blocks):
    size = 1
    for _ in range(num_blocks):
        additional_scope = kernel_size - 1
        for _ in range(num_levels):
            size += additional_scope
            additional_scope *= 2
    return size
