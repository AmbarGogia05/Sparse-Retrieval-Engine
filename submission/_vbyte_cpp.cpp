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

// VByte convention (matches indexer._vbyte_encode): 7 bits per byte, the
// high bit (0x80) is set on the final byte of each number.
static std::vector<std::pair<std::vector<long>, std::vector<long>>>
decode_all(const std::string &data, const std::vector<int> &dfs) {
    std::vector<std::pair<std::vector<long>, std::vector<long>>> out;
    out.reserve(dfs.size());

    const unsigned char *p = reinterpret_cast<const unsigned char *>(data.data());
    size_t pos = 0;

    for (int df : dfs) {
        std::vector<long> ords;
        std::vector<long> tfs;
        ords.reserve(df);
        tfs.reserve(df);

        // Gaps -> absolute ordinals.
        long cur = 0;
        for (int i = 0; i < df; i++) {
            long n = 0;
            int shift = 0;
            unsigned char b;
            do {
                b = p[pos++];
                n |= static_cast<long>(b & 0x7F) << shift;
                shift += 7;
            } while (!(b & 0x80));
            cur = (i == 0) ? n : cur + n;
            ords.push_back(cur);
        }
        // Term frequencies.
        for (int i = 0; i < df; i++) {
            long n = 0;
            int shift = 0;
            unsigned char b;
            do {
                b = p[pos++];
                n |= static_cast<long>(b & 0x7F) << shift;
                shift += 7;
            } while (!(b & 0x80));
            tfs.push_back(n);
        }
        out.emplace_back(std::move(ords), std::move(tfs));
    }
    return out;
}

PYBIND11_MODULE(_vbyte_cpp, m) {
    m.doc() = "Compiled VByte postings decoder (accelerates load_v2).";
    m.def("decode_all", &decode_all,
          "Decode the whole postings blob given per-term dfs -> [(ords, tfs)].");
}
