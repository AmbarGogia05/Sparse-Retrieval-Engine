from submission._index_codec import _decode_tf_blocks, _encode_tf_blocks


def test_adaptive_tf_blocks_round_trip_mixed_values():
    values = (
        [0] * 127
        + [7]
        + [0, 1, 0, 2, 0, 3, 15, 0, 255]
        + [0] * 120
    )

    encoded = _encode_tf_blocks(values)
    decoded, pos = _decode_tf_blocks(encoded, 0, len(values))

    assert decoded == values
    assert pos == len(encoded)


def test_adaptive_tf_blocks_select_rice_for_sparse_nonzero_values():
    encoded = _encode_tf_blocks([0] * 127 + [7])

    assert encoded[0] & 0x80


def test_adaptive_tf_blocks_keep_fixed_width_for_large_single_value():
    encoded = _encode_tf_blocks([10_000])

    assert not encoded[0] & 0x80


def test_tf_decoder_accepts_legacy_fixed_width_block():
    # Four two-bit values, packed LSB-first: 00 01 10 11.
    encoded = bytes([2, 0b11100100])

    decoded, pos = _decode_tf_blocks(encoded, 0, 4)

    assert decoded == [0, 1, 2, 3]
    assert pos == len(encoded)
