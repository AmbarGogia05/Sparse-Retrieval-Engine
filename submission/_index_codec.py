"""Binary codecs shared by the Python index writer and fallback reader.

The C++ builder and native loader implement the same formats. Changes here
must remain byte-compatible with `_spimi_cpp.cpp` and `_index_cpp.cpp`.
"""

from typing import List, Tuple


EF_BLOCK_SIZE = 128
DOC_GAP_BLOCK_SIZE = 32
DOCID_MAGIC = b"D36\x01"
DOCID_PACKED_MAGIC = b"D36\x02"
CASE_TERMS_MAGIC = b"CT1"
TERMS_MAGIC = b"TM2"


def encode_vbyte(numbers: List[int]) -> bytes:
    out = bytearray()
    for number in numbers:
        while True:
            byte = number & 0x7F
            number >>= 7
            if number:
                out.append(byte)
            else:
                out.append(byte | 0x80)
                break
    return bytes(out)


def decode_vbyte(data: bytes, pos: int, count: int) -> Tuple[List[int], int]:
    numbers = []
    for _ in range(count):
        number = 0
        shift = 0
        while True:
            byte = data[pos]
            pos += 1
            number |= (byte & 0x7F) << shift
            if byte & 0x80:
                break
            shift += 7
        numbers.append(number)
    return numbers, pos


def _ef_width(count: int, universe: int) -> int:
    if count <= 0:
        return 0
    quotient = universe // count
    return quotient.bit_length() - 1 if quotient >= 1 else 0


class _BitWriter:
    __slots__ = ("data", "buffer", "bits")

    def __init__(self):
        self.data = bytearray()
        self.buffer = 0
        self.bits = 0

    def write(self, value: int, width: int) -> None:
        self.buffer |= (value & ((1 << width) - 1)) << self.bits
        self.bits += width
        while self.bits >= 8:
            self.data.append(self.buffer & 0xFF)
            self.buffer >>= 8
            self.bits -= 8

    def finish(self) -> bytes:
        if self.bits:
            self.data.append(self.buffer & 0xFF)
            self.buffer = 0
            self.bits = 0
        return bytes(self.data)


class _BitReader:
    __slots__ = ("data", "pos", "buffer", "bits")

    def __init__(self, data: bytes, pos: int):
        self.data = data
        self.pos = pos
        self.buffer = 0
        self.bits = 0

    def read(self, width: int) -> int:
        while self.bits < width:
            self.buffer |= self.data[self.pos] << self.bits
            self.pos += 1
            self.bits += 8
        value = self.buffer & ((1 << width) - 1)
        self.buffer >>= width
        self.bits -= width
        return value

    def read_bit(self) -> int:
        if self.bits == 0:
            self.buffer = self.data[self.pos]
            self.pos += 1
            self.bits = 8
        bit = self.buffer & 1
        self.buffer >>= 1
        self.bits -= 1
        return bit

    def align(self) -> None:
        self.buffer = 0
        self.bits = 0


def _encode_tf_blocks(values: List[int]) -> bytes:
    writer = _BitWriter()
    for start in range(0, len(values), EF_BLOCK_SIZE):
        block = values[start:start + EF_BLOCK_SIZE]
        _write_tf_block(writer, block)
        if writer.bits:
            writer.data.append(writer.buffer & 0xFF)
            writer.buffer = 0
            writer.bits = 0
    return writer.finish()


def _write_tf_block(writer: _BitWriter, block: List[int]) -> None:
    width, use_rice, best_k, _payload_bits = _tf_block_plan(block)

    writer.write((0x80 | best_k) if use_rice else width, 8)
    if use_rice:
        _write_rice_values(writer, block, best_k)
    else:
        for value in block:
            writer.write(value, width)


def _tf_block_plan(block: List[int]) -> Tuple[int, bool, int, int]:
    width = max(block, default=0).bit_length()

    # Header bytes below 0x80 retain the legacy fixed-width format. A header
    # with the high bit set selects Rice coding and stores k in the low five
    # bits. Pick Rice only when its byte-padded payload is strictly smaller, so
    # no block can regress relative to the old codec.
    fixed_bytes = (len(block) * width + 7) // 8
    best_k = 0
    best_rice_bits = None
    for k in range(min(width, 31) + 1):
        rice_bits = sum((value >> k) + 1 + k for value in block)
        if best_rice_bits is None or rice_bits < best_rice_bits:
            best_rice_bits = rice_bits
            best_k = k
    rice_bytes = ((best_rice_bits or 0) + 7) // 8
    use_rice = rice_bytes < fixed_bytes
    payload_bits = (best_rice_bits or 0) if use_rice else len(block) * width
    return width, use_rice, best_k, payload_bits


def _write_rice_values(
    writer: _BitWriter, values: List[int], k: int
) -> None:
    remainder_mask = (1 << k) - 1
    for value in values:
        quotient = value >> k
        while quotient >= 32:
            writer.write(0xFFFFFFFF, 32)
            quotient -= 32
        if quotient:
            writer.write((1 << quotient) - 1, quotient)
        writer.write(0, 1)
        if k:
            writer.write(value & remainder_mask, k)


def _write_vbyte_to_bits(writer: _BitWriter, number: int) -> None:
    for byte in encode_vbyte([number]):
        writer.write(byte, 8)


def _read_vbyte_from_bits(reader: _BitReader) -> int:
    number = 0
    shift = 0
    while True:
        byte = reader.read(8)
        number |= (byte & 0x7F) << shift
        if byte & 0x80:
            return number
        shift += 7


def _read_tf_block(
    reader: _BitReader, block_size: int
) -> List[int]:
    header = reader.read(8)
    if header & 0x80:
        k = header & 0x1F
        values = []
        for _ in range(block_size):
            quotient = 0
            while reader.read_bit():
                quotient += 1
            remainder = reader.read(k) if k else 0
            values.append((quotient << k) | remainder)
        return values

    width = header
    return [reader.read(width) if width else 0 for _ in range(block_size)]


def _decode_tf_blocks(
    data: bytes, pos: int, count: int
) -> Tuple[List[int], int]:
    reader = _BitReader(data, pos)
    values: List[int] = []
    while len(values) < count:
        block_size = min(EF_BLOCK_SIZE, count - len(values))
        values.extend(_read_tf_block(reader, block_size))
        reader.align()
    return values, reader.pos


def encode_postings(
    ordinals: List[int],
    term_frequencies: List[int],
    document_count: int,
    all_tf_one: bool = False,
    packed: bool = False,
    gap_rice: bool = False,
    sparse_tf: bool = False,
    adaptive_gap_rice: bool = False,
) -> bytes:
    count = len(ordinals)
    width = _ef_width(count, document_count)
    mask = (1 << width) - 1

    if packed:
        writer = _BitWriter()
        _write_packed_postings(
            writer, ordinals, term_frequencies, document_count,
            all_tf_one=all_tf_one,
            gap_rice=gap_rice,
            sparse_tf=sparse_tf,
            adaptive_gap_rice=adaptive_gap_rice,
        )
        return writer.finish()

    low_writer = _BitWriter()
    for ordinal in ordinals:
        low_writer.write(ordinal & mask, width)

    high_writer = _BitWriter()
    previous_high = 0
    for ordinal in ordinals:
        high = ordinal >> width
        high_writer.write(0, high - previous_high)
        high_writer.write(1, 1)
        previous_high = high

    encoded = bytearray(low_writer.finish())
    encoded += high_writer.finish()
    if not all_tf_one:
        encoded += _encode_tf_blocks([tf - 1 for tf in term_frequencies])
    return bytes(encoded)


def _write_packed_postings(
    writer: _BitWriter,
    ordinals: List[int],
    term_frequencies: List[int],
    document_count: int,
    all_tf_one: bool = False,
    gap_rice: bool = False,
    sparse_tf: bool = False,
    adaptive_gap_rice: bool = False,
) -> None:
    """Append one term without alignment; TF5 reuses the writer across terms."""
    count = len(ordinals)
    width = _ef_width(count, document_count)
    mask = (1 << width) - 1

    if gap_rice:
        previous = -1
        gaps = []
        for ordinal in ordinals:
            gaps.append(ordinal - previous - 1)
            previous = ordinal
        if adaptive_gap_rice:
            for start in range(0, len(gaps), DOC_GAP_BLOCK_SIZE):
                block = gaps[start:start + DOC_GAP_BLOCK_SIZE]
                global_bits = sum(
                    (value >> width) + 1 + width for value in block
                )
                mean_width = min(
                    (sum(block) // len(block)).bit_length() - 1,
                    31,
                )
                mean_width = max(mean_width, 0)
                candidates = {
                    max(mean_width - 1, 0),
                    mean_width,
                    min(mean_width + 1, 31),
                }
                costs = [
                    (
                        sum((value >> k) + 1 + k for value in block),
                        k,
                    )
                    for k in candidates
                ]
                best_bits, best_k = min(costs)
                delta = best_k - width
                header_bits = 4 if -3 <= delta <= 3 else 9
                if best_bits + header_bits < global_bits + 1:
                    writer.write(1, 1)
                    if -3 <= delta <= 3:
                        writer.write(delta + 3, 3)
                    else:
                        writer.write(7, 3)
                        writer.write(best_k, 5)
                    _write_rice_values(writer, block, best_k)
                else:
                    writer.write(0, 1)
                    _write_rice_values(writer, block, width)
        else:
            _write_rice_values(writer, gaps, width)
    else:
        for ordinal in ordinals:
            writer.write(ordinal & mask, width)

        previous_high = 0
        for ordinal in ordinals:
            high = ordinal >> width
            writer.write(0, high - previous_high)
            writer.write(1, 1)
            previous_high = high

    if sparse_tf:
        positions = [
            index for index, tf in enumerate(term_frequencies)
            if tf > 1
        ]
        _write_vbyte_to_bits(writer, len(positions))
        position_width = _ef_width(len(positions), count)
        previous = -1
        gaps = []
        for position in positions:
            gaps.append(position - previous - 1)
            previous = position
        _write_rice_values(writer, gaps, position_width)
        excess = [
            tf - 2 for tf in term_frequencies if tf > 1
        ]
        for start in range(0, len(excess), EF_BLOCK_SIZE):
            _write_tf_block(
                writer, excess[start:start + EF_BLOCK_SIZE]
            )
    elif not all_tf_one:
        values = [tf - 1 for tf in term_frequencies]
        for start in range(0, len(values), EF_BLOCK_SIZE):
            _write_tf_block(
                writer, values[start:start + EF_BLOCK_SIZE]
            )


def _decode_packed_postings(
    reader: _BitReader,
    document_frequency: int,
    document_count: int,
    all_tf_one: bool = False,
    gap_rice: bool = False,
    sparse_tf: bool = False,
    adaptive_gap_rice: bool = False,
) -> Tuple[List[int], List[int]]:
    count = document_frequency
    width = _ef_width(count, document_count)
    if gap_rice:
        ordinals = []
        previous = -1
        done = 0
        while done < count:
            block_size = min(DOC_GAP_BLOCK_SIZE, count - done)
            rice_k = width
            if adaptive_gap_rice and reader.read_bit():
                code = reader.read(3)
                rice_k = (
                    reader.read(5) if code == 7
                    else width + code - 3
                )
            for _ in range(block_size):
                quotient = 0
                while reader.read_bit():
                    quotient += 1
                remainder = reader.read(rice_k) if rice_k else 0
                previous += ((quotient << rice_k) | remainder) + 1
                ordinals.append(previous)
            done += block_size
    else:
        low_bits = [
            reader.read(width) if width else 0 for _ in range(count)
        ]

        ordinals = []
        high = 0
        while len(ordinals) < count:
            if reader.read_bit():
                ordinals.append(
                    (high << width) | low_bits[len(ordinals)]
                )
            else:
                high += 1

    if all_tf_one:
        term_frequencies = [1] * count
    elif sparse_tf:
        exception_count = _read_vbyte_from_bits(reader)
        position_width = _ef_width(exception_count, count)
        positions = []
        previous = -1
        for _ in range(exception_count):
            quotient = 0
            while reader.read_bit():
                quotient += 1
            remainder = (
                reader.read(position_width) if position_width else 0
            )
            previous += (
                (quotient << position_width) | remainder
            ) + 1
            positions.append(previous)

        excess: List[int] = []
        while len(excess) < exception_count:
            block_size = min(
                EF_BLOCK_SIZE, exception_count - len(excess)
            )
            excess.extend(_read_tf_block(reader, block_size))
        term_frequencies = [1] * count
        for position, value in zip(positions, excess):
            term_frequencies[position] = value + 2
    else:
        tf_minus_one: List[int] = []
        while len(tf_minus_one) < count:
            block_size = min(
                EF_BLOCK_SIZE, count - len(tf_minus_one)
            )
            tf_minus_one.extend(_read_tf_block(reader, block_size))
        term_frequencies = [value + 1 for value in tf_minus_one]
    return ordinals, term_frequencies


def decode_postings(
    data: bytes,
    pos: int,
    document_frequency: int,
    document_count: int,
    all_tf_one: bool = False,
    packed: bool = False,
    gap_rice: bool = False,
    sparse_tf: bool = False,
    adaptive_gap_rice: bool = False,
) -> Tuple[List[int], List[int], int]:
    count = document_frequency
    width = _ef_width(count, document_count)

    if packed:
        reader = _BitReader(data, pos)
        ordinals, term_frequencies = _decode_packed_postings(
            reader, count, document_count,
            all_tf_one=all_tf_one,
            gap_rice=gap_rice,
            sparse_tf=sparse_tf,
            adaptive_gap_rice=adaptive_gap_rice,
        )
        reader.align()
        return ordinals, term_frequencies, reader.pos

    low_reader = _BitReader(data, pos)
    low_bits = [low_reader.read(width) if width else 0 for _ in range(count)]
    pos += (count * width + 7) // 8

    high_reader = _BitReader(data, pos)
    ordinals: List[int] = []
    high = 0
    while len(ordinals) < count:
        if high_reader.read_bit():
            ordinals.append((high << width) | low_bits[len(ordinals)])
        else:
            high += 1
    high_reader.align()

    if all_tf_one:
        return ordinals, [1] * count, high_reader.pos

    tf_minus_one, pos = _decode_tf_blocks(data, high_reader.pos, count)
    return ordinals, [value + 1 for value in tf_minus_one], pos


def encode_terms(term_metadata: List[Tuple[str, int]]) -> bytes:
    out = bytearray(TERMS_MAGIC)
    out += encode_vbyte([len(term_metadata)])
    previous = ""
    for term, document_frequency in term_metadata:
        shared = 0
        limit = min(len(previous), len(term))
        while shared < limit and previous[shared] == term[shared]:
            shared += 1
        suffix = term[shared:].encode("utf-8")
        out += encode_vbyte([shared])
        out += suffix
        out.append(0)
        previous = term
    out += encode_vbyte([
        document_frequency for _, document_frequency in term_metadata
    ])
    return bytes(out)


def decode_terms(data: bytes) -> Tuple[List[str], List[int]]:
    if data.startswith(TERMS_MAGIC):
        (count,), pos = decode_vbyte(data, len(TERMS_MAGIC), 1)
        terms: List[str] = []
        previous = ""
        for _ in range(count):
            (shared,), pos = decode_vbyte(data, pos, 1)
            end = data.index(0, pos)
            suffix = data[pos:end].decode("utf-8")
            pos = end + 1
            term = previous[:shared] + suffix
            terms.append(term)
            previous = term
        document_frequencies, pos = decode_vbyte(data, pos, count)
        return terms, document_frequencies

    terms: List[str] = []
    document_frequencies: List[int] = []
    pos = 0
    previous = ""
    while pos < len(data):
        (shared, suffix_length), pos = decode_vbyte(data, pos, 2)
        suffix = data[pos:pos + suffix_length].decode("utf-8")
        pos += suffix_length
        (document_frequency,), pos = decode_vbyte(data, pos, 1)
        term = previous[:shared] + suffix
        terms.append(term)
        document_frequencies.append(document_frequency)
        previous = term
    return terms, document_frequencies


def encode_case_terms(terms: set) -> bytes:
    out = bytearray(CASE_TERMS_MAGIC)
    previous = ""
    for term in sorted(terms):
        shared = 0
        limit = min(len(previous), len(term))
        while shared < limit and previous[shared] == term[shared]:
            shared += 1
        suffix = term[shared:].encode("utf-8")
        out += encode_vbyte([shared, len(suffix)])
        out += suffix
        previous = term
    return bytes(out)


def decode_case_terms(data: bytes) -> set:
    if not data.startswith(CASE_TERMS_MAGIC):
        return set()

    terms = set()
    pos = len(CASE_TERMS_MAGIC)
    previous = ""
    while pos < len(data):
        (shared, suffix_length), pos = decode_vbyte(data, pos, 2)
        suffix = data[pos:pos + suffix_length].decode("utf-8")
        pos += suffix_length
        term = previous[:shared] + suffix
        terms.add(term)
        previous = term
    return terms


def encode_docid(doc_id: str):
    if len(doc_id) != 8 or any(
        char not in "0123456789abcdefghijklmnopqrstuvwxyz" for char in doc_id
    ):
        return None

    value = 0
    for char in doc_id:
        digit = ord(char) - 48 if char <= "9" else ord(char) - 87
        value = value * 36 + digit
    return value.to_bytes(6, "little")


def encode_docids(doc_ids: List[str]):
    writer = _BitWriter()
    for doc_id in doc_ids:
        encoded = encode_docid(doc_id)
        if encoded is None:
            return None
        writer.write(int.from_bytes(encoded, "little"), 42)
    return DOCID_PACKED_MAGIC + writer.finish()


def decode_docids(data: bytes) -> List[str]:
    values = []
    if data.startswith(DOCID_PACKED_MAGIC):
        count = ((len(data) - len(DOCID_PACKED_MAGIC)) * 8) // 42
        reader = _BitReader(data, len(DOCID_PACKED_MAGIC))
        values = [reader.read(42) for _ in range(count)]
    elif data.startswith(DOCID_MAGIC):
        values = [
            int.from_bytes(data[pos:pos + 6], "little")
            for pos in range(len(DOCID_MAGIC), len(data), 6)
        ]
    else:
        return []

    doc_ids = []
    for value in values:
        characters = ["0"] * 8
        for index in range(7, -1, -1):
            digit = value % 36
            characters[index] = (
                chr(48 + digit) if digit < 10 else chr(87 + digit)
            )
            value //= 36
        doc_ids.append("".join(characters))
    return doc_ids
