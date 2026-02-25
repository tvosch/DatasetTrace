/**
 * cpp_engine.h — FM-index query engine (header-only implementation).
 *
 * Wraps SDSL's compressed suffix arrays (csa_wt) to support three operations:
 *   find()            — backward search; returns the SA interval for a query string.
 *   count()           — number of occurrences of a query string in the corpus.
 *   get_doc_by_rank() — retrieve the document at a given SA rank, with context.
 *
 * Each Engine instance manages one or more FMIndexShards.  Shards are searched
 * in parallel (one std::thread per shard) and results are aggregated.
 *
 * Compile with:
 *   c++ -std=c++17 -O3 -shared -fPIC \
 *       $(python3 -m pybind11 --includes) \
 *       src/cpp_engine.cpp \
 *       -o src/cpp_engine$(python3-config --extension-suffix) \
 *       -I../../sdsl/include -L../../sdsl/lib \
 *       -lsdsl -ldivsufsort -ldivsufsort64 -pthread
 *   (run from infini-gram-mini/infini-gram-mini/engine/)
 */

#pragma once

#include <sdsl/suffix_arrays.hpp>

#include <cassert>
#include <cstddef>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

// FM-index type: wavelet-tree compressed suffix array with RRR-compressed BWT.
// Sampling densities (32, 64) trade off SA/ISA lookup speed vs. index size.
using index_t = sdsl::csa_wt<sdsl::wt_huff<sdsl::rrr_vector<127>>, 32, 64>;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/**
 * All data for one index shard.
 *
 * data_index / meta_index are heap-allocated SDSL objects whose internal
 * memory is managed by SDSL itself (either heap or mmap, depending on how
 * they were loaded).  They must always be released with `delete`.
 *
 * data_offset / meta_offset are memory-mapped arrays of uint64 values, one
 * per document, storing the byte offset of each document start in the corpus.
 * They must be released with munmap(ptr, size_bytes).
 */
struct FMIndexShard {
    index_t*  data_index;
    std::size_t*   data_offset;
    std::size_t    data_offset_size;  ///< bytes mapped for data_offset (for munmap)

    index_t*  meta_index;             ///< null if get_metadata == false
    std::size_t*   meta_offset;       ///< null if get_metadata == false
    std::size_t    meta_offset_size;  ///< bytes mapped for meta_offset (for munmap)

    std::size_t    doc_cnt;           ///< number of documents in this shard
};

/**
 * Result of a find() query.
 * segment_by_shard[s] = [lo, hi): half-open interval in the SA of shard s.
 */
struct FindResult {
    std::size_t cnt;
    std::vector<std::pair<std::size_t, std::size_t>> segment_by_shard;
};

/** Result of a count() query. */
struct CountResult {
    std::size_t count;
};

/** Result of a get_doc_by_rank() query. */
struct DocResult {
    std::size_t doc_ix;       ///< global (cross-shard) document index
    std::size_t doc_len;      ///< full document length in bytes
    std::size_t disp_len;     ///< bytes in the displayed context window
    std::size_t needle_offset;///< byte offset of the query within the context window
    std::string metadata;     ///< JSON string with source path + per-doc fields
    std::string text;         ///< UTF-8 text of the context window
};

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

class Engine {
public:

    /**
     * Load one or more index shards and prepare for querying.
     *
     * @param index_dirs   Paths to shard directories (each must contain data.fm9,
     *                     data_offset, and optionally meta.fm9 / meta_offset).
     * @param load_to_ram  If true, copy the FM-index into RAM; otherwise use mmap.
     * @param get_metadata If true, also load the metadata index.
     */
    Engine(const std::vector<std::string> index_dirs, bool load_to_ram, bool get_metadata)
        : _load_to_ram(load_to_ram), _get_metadata(get_metadata)
    {
        for (const auto& index_dir : index_dirs) {
            assert(fs::exists(index_dir));

            // --- Data FM-index ---
            auto* data_index = new index_t();
            const std::string data_index_path = index_dir + "/data.fm9";
            if (_load_to_ram) {
                sdsl::load_from_file(*data_index, data_index_path);
            } else {
                sdsl::load_from_file_(*data_index, data_index_path);
            }

            // --- Data offset table (mmap'd uint64 array, one entry per document) ---
            const std::string data_offset_path = index_dir + "/data_offset";
            int data_fd = open(data_offset_path.c_str(), O_RDONLY);
            assert(data_fd >= 0);
            const off_t data_offset_size = lseek(data_fd, 0, SEEK_END);
            assert(data_offset_size > 0);
            auto* data_offset = static_cast<std::size_t*>(
                mmap(nullptr, data_offset_size, PROT_READ, MAP_PRIVATE, data_fd, 0));
            assert(data_offset != MAP_FAILED);
            close(data_fd);  // fd can be closed once mmap is established

            const std::size_t doc_cnt = data_offset_size / sizeof(std::size_t);

            // --- Metadata FM-index (optional) ---
            index_t*     meta_index       = nullptr;
            std::size_t* meta_offset      = nullptr;
            std::size_t  meta_offset_size = 0;

            if (_get_metadata) {
                meta_index = new index_t();
                const std::string meta_index_path = index_dir + "/meta.fm9";
                if (_load_to_ram) {
                    sdsl::load_from_file(*meta_index, meta_index_path);
                } else {
                    sdsl::load_from_file_(*meta_index, meta_index_path);
                }

                const std::string meta_offset_path = index_dir + "/meta_offset";
                int meta_fd = open(meta_offset_path.c_str(), O_RDONLY);
                assert(meta_fd >= 0);
                meta_offset_size = lseek(meta_fd, 0, SEEK_END);
                assert(meta_offset_size > 0);
                meta_offset = static_cast<std::size_t*>(
                    mmap(nullptr, meta_offset_size, PROT_READ, MAP_PRIVATE, meta_fd, 0));
                assert(meta_offset != MAP_FAILED);
                close(meta_fd);
            }

            _shards.push_back(FMIndexShard{
                data_index, data_offset, static_cast<std::size_t>(data_offset_size),
                meta_index, meta_offset, meta_offset_size,
                doc_cnt,
            });
        }

        _num_shards = _shards.size();
        assert(_num_shards > 0);
    }

    ~Engine() {
        for (auto& shard : _shards) {
            // In RAM mode the SDSL objects own heap memory; delete is correct.
            // In mmap mode load_from_file_() backs the SDSL internals with mmap'd
            // regions.  Calling delete would invoke SDSL's destructor which tries
            // to free() those regions, causing "free(): invalid pointer".  We skip
            // delete in that case and let the OS reclaim the mmap on process exit.
            if (_load_to_ram) {
                delete shard.data_index;
            }

            munmap(shard.data_offset, shard.data_offset_size);

            if (_get_metadata) {
                if (_load_to_ram) {
                    delete shard.meta_index;
                }
                munmap(shard.meta_offset, shard.meta_offset_size);
            }
        }
    }

    // Disable copy (shards hold raw pointers / mmap regions).
    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    // ------------------------------------------------------------------
    // Public query API
    // ------------------------------------------------------------------

    /**
     * Find the suffix-array interval [lo, hi) for each shard where *query* occurs.
     * Shards are searched in parallel.  An empty query matches the whole corpus.
     */
    FindResult find(const std::string query) const {
        std::vector<std::pair<std::size_t, std::size_t>> segment_by_shard(_num_shards);

        if (query.empty()) {
            // Empty query matches everything; interval spans the entire SA.
            for (std::size_t s = 0; s < _num_shards; ++s) {
                segment_by_shard[s] = {0, _shards[s].data_index->size()};
            }
        } else {
            std::vector<std::thread> threads;
            threads.reserve(_num_shards);
            for (std::size_t s = 0; s < _num_shards; ++s) {
                threads.emplace_back(&Engine::_find_thread, this, s, &query, &segment_by_shard[s]);
            }
            for (auto& t : threads) {
                t.join();
            }
        }

        std::size_t cnt = 0;
        for (std::size_t s = 0; s < _num_shards; ++s) {
            assert(segment_by_shard[s].first <= segment_by_shard[s].second);
            cnt += segment_by_shard[s].second - segment_by_shard[s].first;
        }

        return FindResult{.cnt = cnt, .segment_by_shard = segment_by_shard};
    }

    /** Count occurrences of *query* across the entire corpus. */
    CountResult count(const std::string& query) const {
        return CountResult{.count = find(query).cnt};
    }

    /**
     * Retrieve the document that contains the occurrence at position *rank*
     * in the suffix array of shard *s*, together with a context window.
     *
     * @param s           Shard index (must be < num_shards).
     * @param rank        Position in the suffix array of shard s.
     * @param needle_len  Byte length of the query string.
     * @param max_ctx_len Maximum context bytes on each side of the needle.
     */
    DocResult get_doc_by_rank(
        const std::size_t s,
        const std::size_t rank,
        const std::size_t needle_len,
        const std::size_t max_ctx_len) const
    {
        assert(s < _num_shards);
        const auto& shard = _shards[s];
        assert(rank < shard.data_index->size());

        // Map the SA rank to a corpus byte position.
        const std::size_t ptr = (*shard.data_index)[rank];

        // Binary search over the document offset table to find which document
        // contains byte position ptr.
        std::size_t lo = 0, hi = shard.doc_cnt;
        while (hi - lo > 1) {
            std::size_t mid = (lo + hi) >> 1;
            if (_doc_ix_to_ptr(shard, mid) <= ptr) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        const std::size_t local_doc_ix = lo;

        // Compute global document index by summing doc counts from earlier shards.
        std::size_t doc_ix = local_doc_ix;
        for (std::size_t i = 0; i < s; ++i) {
            doc_ix += _shards[i].doc_cnt;
        }

        // Document byte range (skip the leading document-separator byte with +1).
        const std::size_t doc_start = _doc_ix_to_ptr(shard, local_doc_ix) + 1;
        const std::size_t doc_end   = _doc_ix_to_ptr(shard, local_doc_ix + 1);
        const std::size_t doc_len   = doc_end - doc_start;

        // Context window: up to max_ctx_len bytes on each side of the needle.
        const std::size_t disp_start  = std::max(doc_start, ptr < max_ctx_len ? 0UL : ptr - max_ctx_len);
        const std::size_t disp_end    = std::min(doc_end,   ptr + needle_len + max_ctx_len);
        const std::size_t disp_len    = disp_end - disp_start;
        const std::size_t needle_off  = ptr - disp_start;

        std::string text;
        if (disp_start < disp_end) {
            text = _parallel_extract(s, disp_start, disp_end, /*is_meta=*/false);
        }

        std::string metadata;
        if (_get_metadata) {
            const std::size_t meta_start = _doc_ix_to_meta_ptr(shard, local_doc_ix);
            // -1: exclude trailing '\n' written by the indexer
            const std::size_t meta_end   = _doc_ix_to_meta_ptr(shard, local_doc_ix + 1) - 1;
            if (meta_start < meta_end) {
                metadata = _parallel_extract(s, meta_start, meta_end, /*is_meta=*/true);
            }
        }

        return DocResult{
            .doc_ix        = doc_ix,
            .doc_len       = doc_len,
            .disp_len      = disp_len,
            .needle_offset = needle_off,
            .metadata      = metadata,
            .text          = text,
        };
    }

private:

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /** Worker thread: run backward search for *query* in shard *s*. */
    void _find_thread(
        const std::size_t s,
        const std::string* const query,
        std::pair<std::size_t, std::size_t>* const segment) const
    {
        std::size_t lo = 0, hi = 0;
        sdsl::backward_search(
            *_shards[s].data_index,
            0, _shards[s].data_index->size() - 1,
            query->begin(), query->end(),
            lo, hi);
        segment->first  = lo;
        segment->second = hi + 1;  // convert to half-open [lo, hi)
    }

    /**
     * Extract bytes [start, end) from the FM-index of shard *s*.
     * For ranges >= 100 bytes, spawns up to 10 threads to extract in parallel.
     */
    std::string _parallel_extract(
        std::size_t shard_ix,
        std::size_t start,
        std::size_t end,
        bool is_meta) const
    {
        if (start >= end) return "";

        const std::size_t total = end - start;

        // Below this threshold, single-threaded extraction is faster.
        if (total < 100) {
            return is_meta
                ? sdsl::extract(*_shards[shard_ix].meta_index, start, end - 1)
                : sdsl::extract(*_shards[shard_ix].data_index, start, end - 1);
        }

        const std::size_t num_threads  = std::min(total / 100, std::size_t{10});
        const std::size_t chunk        = (total + num_threads - 1) / num_threads;

        std::vector<std::string> segments(num_threads);
        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        for (std::size_t i = 0; i < num_threads; ++i) {
            const std::size_t chunk_start = start + i * chunk;
            if (chunk_start >= end) {
                break;
            }
            const std::size_t chunk_end = std::min(chunk_start + chunk, end);
            threads.emplace_back(
                &Engine::_extract_thread, this,
                shard_ix, chunk_start, chunk_end, &segments[i], is_meta);
        }
        for (auto& t : threads) {
            t.join();
        }

        std::string result;
        result.reserve(total);
        for (auto& seg : segments) {
            result += seg;
        }
        return result;
    }

    /** Worker thread: extract bytes [start, end) from the FM-index. */
    void _extract_thread(
        std::size_t shard_ix,
        std::size_t start,
        std::size_t end,
        std::string* out,
        bool is_meta) const
    {
        *out = is_meta
            ? sdsl::extract(*_shards[shard_ix].meta_index, start, end - 1)
            : sdsl::extract(*_shards[shard_ix].data_index, start, end - 1);
    }

    /**
     * Return the corpus byte offset of document *doc_ix* in *shard*.
     * For doc_ix == doc_cnt, returns the last valid corpus position.
     */
    inline std::size_t _doc_ix_to_ptr(
        const FMIndexShard& shard, const std::size_t doc_ix) const
    {
        assert(doc_ix <= shard.doc_cnt);
        if (doc_ix == shard.doc_cnt) {
            return shard.data_index->size() - 1;  // exclude terminal '\0'
        }
        return shard.data_offset[doc_ix];
    }

    /**
     * Return the metadata byte offset of document *doc_ix* in *shard*.
     * For doc_ix == doc_cnt, returns the last valid metadata position.
     */
    inline std::size_t _doc_ix_to_meta_ptr(
        const FMIndexShard& shard, const std::size_t doc_ix) const
    {
        assert(doc_ix <= shard.doc_cnt);
        if (doc_ix == shard.doc_cnt) {
            return shard.meta_index->size() - 1;  // exclude terminal '\0'
        }
        return shard.meta_offset[doc_ix];
    }

    // ------------------------------------------------------------------
    // Members
    // ------------------------------------------------------------------

    std::vector<FMIndexShard> _shards;
    std::size_t               _num_shards;
    bool                      _load_to_ram;
    bool                      _get_metadata;
};
