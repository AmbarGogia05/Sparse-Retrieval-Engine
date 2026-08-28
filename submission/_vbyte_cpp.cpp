// submission/_vbyte_cpp.cpp — optional compiled acceleration for the
// compressed-index load path (InvertedIndex.load_v2 in indexer.py).
//
// decode_all() takes the whole postings.bin blob plus the per-term document
// frequencies (in block order) and returns, for each term, its absolute
// sorted doc ordinals and term frequencies. Doing the VByte decode and the
// gap->absolute conversion here — in one call over the whole blob — avoids
// hundreds of thousands of Python<->C++ boundary crossings.
//
// This is a drop-in accelerator: indexer.py imports it if built and falls
// back to a pure-Python decoder otherwise, so nothing breaks if the
// extension is not compiled on the target machine.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <utility>

namespace py = pybind11;

static long read_vbyte(const unsigned char *p, size_t &pos) {
    long n = 0;
    int shift = 0;
    unsigned char b;
    do {
        b = p[pos++];
        n |= static_cast<long>(b & 0x7F) << shift;
        shift += 7;
    } while (!(b & 0x80));
    return n;
}

// VByte convention (matches indexer._vbyte_encode): 7 bits per byte, the
// high bit (0x80) is set on the final byte of each number.
//
// A1 codec (matches indexer._encode_postings / _spimi_cpp.cpp's
// put_postings): a posting list is `df` codes (`gap*2 + (tf>1 ? 1 : 0)`)
// followed by ONLY the tfs for postings whose flag bit is set — a tf==1
// posting costs zero tf bytes.
static std::vector<std::pair<std::vector<long>, std::vector<long>>>
decode_all(const std::string &data, const std::vector<int> &dfs) {
    std::vector<std::pair<std::vector<long>, std::vector<long>>> out;
    out.reserve(dfs.size());

    const unsigned char *p = reinterpret_cast<const unsigned char *>(data.data());
    size_t pos = 0;

    for (int df : dfs) {
        std::vector<long> ords;
        std::vector<long> tfs;
        std::vector<unsigned char> flags;
        ords.reserve(df);
        tfs.reserve(df);
        flags.reserve(df);

        // Codes -> absolute ordinals + flag bits.
        long cur = 0;
        for (int i = 0; i < df; i++) {
            long code = read_vbyte(p, pos);
            long gap = code >> 1;
            flags.push_back(static_cast<unsigned char>(code & 1));
            cur = (i == 0) ? gap : cur + gap;
            ords.push_back(cur);
        }
        // Term frequencies: only for flagged postings; tf==1 elsewhere.
        for (int i = 0; i < df; i++)
            tfs.push_back(flags[i] ? read_vbyte(p, pos) : 1L);
        out.emplace_back(std::move(ords), std::move(tfs));
    }
    return out;
}

PYBIND11_MODULE(_vbyte_cpp, m) {
    m.doc() = "Compiled VByte postings decoder (accelerates load_v2).";
    m.def("decode_all", &decode_all,
          "Decode the whole postings blob given per-term dfs -> [(ords, tfs)].");
}
