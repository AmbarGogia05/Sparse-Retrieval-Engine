"""Binary codecs shared by the Python index writer and fallback reader.

The C++ builder and native loader implement the same formats. Changes here
must remain byte-compatible with `_spimi_cpp.cpp` and `_index_cpp.cpp`.
"""

from typing import List, Tuple


EF_BLOCK_SIZE = 128
DOCID_MAGIC = b"D36\x01"
CASE_TERMS_MAGIC = b"CT1"


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
        width = max(block, default=0).bit_length()
        writer.write(width, 8)
        for value in block:
            writer.write(value, width)
        if writer.bits:
            writer.data.append(writer.buffer & 0xFF)
            writer.buffer = 0
            writer.bits = 0
    return writer.finish()


def _decode_tf_blocks(
    data: bytes, pos: int, count: int
) -> Tuple[List[int], int]:
    reader = _BitReader(data, pos)
    values: List[int] = []
    while len(values) < count:
        block_size = min(EF_BLOCK_SIZE, count - len(values))
        width = reader.read(8)
        values.extend(
            reader.read(width) if width else 0 for _ in range(block_size)
        )
        reader.align()
    return values, reader.pos


def encode_postings(
    ordinals: List[int],
    term_frequencies: List[int],
    document_count: int,
    all_tf_one: bool = False,
) -> bytes:
    count = len(ordinals)
    width = _ef_width(count, document_count)
    mask = (1 << width) - 1

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


def decode_postings(
    data: bytes,
    pos: int,
    document_frequency: int,
    document_count: int,
    all_tf_one: bool = False,
) -> Tuple[List[int], List[int], int]:
    count = document_frequency
    width = _ef_width(count, document_count)

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
    out = bytearray()
    previous = ""
    for term, document_frequency in term_metadata:
        shared = 0
        limit = min(len(previous), len(term))
        while shared < limit and previous[shared] == term[shared]:
            shared += 1
        suffix = term[shared:].encode("utf-8")
        out += encode_vbyte([shared, len(suffix)])
        out += suffix
        out += encode_vbyte([document_frequency])
        previous = term
    return bytes(out)


def decode_terms(data: bytes) -> Tuple[List[str], List[int]]:
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


def decode_docids(data: bytes) -> List[str]:
    if not data.startswith(DOCID_MAGIC):
        return []

    doc_ids = []
    for pos in range(len(DOCID_MAGIC), len(data), 6):
        value = int.from_bytes(data[pos:pos + 6], "little")
        characters = ["0"] * 8
        for index in range(7, -1, -1):
            digit = value % 36
            characters[index] = (
                chr(48 + digit) if digit < 10 else chr(87 + digit)
            )
            value //= 36
        doc_ids.append("".join(characters))
    return doc_ids
