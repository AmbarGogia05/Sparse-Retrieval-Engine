from submission._index_codec import (
    _BitReader,
    _BitWriter,
    _decode_packed_postings,
    _decode_tf_blocks,
    _encode_tf_blocks,
    _write_packed_postings,
    decode_docids,
    decode_postings,
    decode_terms,
    encode_docids,
    encode_postings,
    encode_terms,
    encode_vbyte,
)


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


def test_grouped_term_metadata_round_trip():
    metadata = [
        ("alpha", 1),
        ("alphabet", 127),
        ("alpine", 128),
        ("βeta", 20_000),
    ]

    encoded = encode_terms(metadata)

    assert encoded.startswith(b"TM2")
    assert decode_terms(encoded) == (
        [term for term, _ in metadata],
        [df for _, df in metadata],
    )


def test_term_decoder_accepts_legacy_interleaved_metadata():
    encoded = bytearray()
    previous = ""
    metadata = [("alpha", 3), ("alphabet", 200), ("beta", 1)]
    for term, df in metadata:
        shared = 0
        while (
            shared < min(len(previous), len(term))
            and previous[shared] == term[shared]
        ):
            shared += 1
        suffix = term[shared:].encode()
        encoded += encode_vbyte([shared, len(suffix)])
        encoded += suffix
        encoded += encode_vbyte([df])
        previous = term

    assert decode_terms(bytes(encoded)) == (
        [term for term, _ in metadata],
        [df for _, df in metadata],
    )


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


def test_continuously_packed_postings_round_trip():
    ordinals = [1, 4, 8, 15, 16, 31, 63, 95, 127, 191]
    term_frequencies = [1, 2, 1, 4, 1, 8, 2, 1, 3, 1]

    encoded = encode_postings(
        ordinals,
        term_frequencies,
        document_count=256,
        packed=True,
    )
    decoded_ordinals, decoded_tfs, pos = decode_postings(
        encoded,
        0,
        len(ordinals),
        document_count=256,
        packed=True,
    )

    assert decoded_ordinals == ordinals
    assert decoded_tfs == term_frequencies
    assert pos == len(encoded)


def test_continuously_packed_tf1_postings_round_trip():
    ordinals = [0, 2, 7, 13, 31, 63]

    encoded = encode_postings(
        ordinals,
        [1] * len(ordinals),
        document_count=128,
        all_tf_one=True,
        packed=True,
    )
    decoded_ordinals, decoded_tfs, pos = decode_postings(
        encoded,
        0,
        len(ordinals),
        document_count=128,
        all_tf_one=True,
        packed=True,
    )

    assert decoded_ordinals == ordinals
    assert decoded_tfs == [1] * len(ordinals)
    assert pos == len(encoded)


def test_gap_rice_postings_round_trip():
    ordinals = [0, 1, 3, 17, 18, 65, 127, 128, 255]
    term_frequencies = [1, 4, 2, 1, 8, 1, 3, 2, 1]

    encoded = encode_postings(
        ordinals,
        term_frequencies,
        document_count=512,
        packed=True,
        gap_rice=True,
    )
    decoded_ordinals, decoded_tfs, pos = decode_postings(
        encoded,
        0,
        len(ordinals),
        document_count=512,
        packed=True,
        gap_rice=True,
    )

    assert decoded_ordinals == ordinals
    assert decoded_tfs == term_frequencies
    assert pos == len(encoded)


def test_block_adaptive_gap_rice_postings_round_trip():
    ordinals = (
        list(range(32))
        + list(range(1000, 1032))
        + [2000, 4000, 8000]
    )
    term_frequencies = [1] * len(ordinals)

    encoded = encode_postings(
        ordinals,
        term_frequencies,
        document_count=10_000,
        all_tf_one=True,
        packed=True,
        gap_rice=True,
        adaptive_gap_rice=True,
    )
    decoded_ordinals, decoded_tfs, pos = decode_postings(
        encoded,
        0,
        len(ordinals),
        document_count=10_000,
        all_tf_one=True,
        packed=True,
        gap_rice=True,
        adaptive_gap_rice=True,
    )

    assert decoded_ordinals == ordinals
    assert decoded_tfs == term_frequencies
    assert pos == len(encoded)


def test_42_bit_docids_round_trip():
    doc_ids = ["00000000", "ug7v899j", "zzzzzzzz"]

    encoded = encode_docids(doc_ids)

    assert encoded is not None
    assert encoded.startswith(b"D36\x02")
    assert len(encoded) == 4 + (42 * len(doc_ids) + 7) // 8
    assert decode_docids(encoded) == doc_ids


def test_sparse_tf_exception_postings_round_trip():
    ordinals = list(range(0, 512, 4))
    term_frequencies = [1] * len(ordinals)
    term_frequencies[3] = 2
    term_frequencies[75] = 7

    encoded = encode_postings(
        ordinals,
        term_frequencies,
        document_count=512,
        packed=True,
        gap_rice=True,
        sparse_tf=True,
    )
    decoded_ordinals, decoded_tfs, pos = decode_postings(
        encoded,
        0,
        len(ordinals),
        document_count=512,
        packed=True,
        gap_rice=True,
        sparse_tf=True,
    )

    assert decoded_ordinals == ordinals
    assert decoded_tfs == term_frequencies
    assert pos == len(encoded)


def test_continuous_postings_stream_round_trip():
    records = [
        ([1, 5, 9], [1, 1, 1], True),
        ([0, 4, 8, 12, 31], [1, 3, 1, 2, 1], False),
    ]
    writer = _BitWriter()
    for ordinals, term_frequencies, all_one in records:
        _write_packed_postings(
            writer,
            ordinals,
            term_frequencies,
            document_count=64,
            all_tf_one=all_one,
            gap_rice=True,
            sparse_tf=not all_one,
        )
    encoded = writer.finish()

    reader = _BitReader(encoded, 0)
    decoded = []
    for ordinals, _term_frequencies, all_one in records:
        decoded.append(
            _decode_packed_postings(
                reader,
                len(ordinals),
                document_count=64,
                all_tf_one=all_one,
                gap_rice=True,
                sparse_tf=not all_one,
            )
        )

    assert decoded == [
        (record[0], record[1]) for record in records
    ]
    assert reader.pos == len(encoded)
