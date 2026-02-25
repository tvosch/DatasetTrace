/**
 * cpp_engine.cpp — pybind11 bindings for the FM-index query engine.
 *
 * Compile from infini-gram-mini/infini-gram-mini/engine/:
 *   c++ -std=c++17 -O3 -shared -fPIC \
 *       $(python3 -m pybind11 --includes) \
 *       src/cpp_engine.cpp \
 *       -o src/cpp_engine$(python3-config --extension-suffix) \
 *       -I../../sdsl/include -L../../sdsl/lib \
 *       -lsdsl -ldivsufsort -ldivsufsort64 -pthread
 */

#include "cpp_engine.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(cpp_engine, m) {
    m.doc() = "FM-index query engine (C++ backend).";

    py::class_<FindResult>(m, "FindResult")
        .def_readwrite("cnt",              &FindResult::cnt)
        .def_readwrite("segment_by_shard", &FindResult::segment_by_shard);

    py::class_<CountResult>(m, "CountResult")
        .def_readwrite("count", &CountResult::count);

    py::class_<DocResult>(m, "DocResult")
        .def_readwrite("doc_ix",        &DocResult::doc_ix)
        .def_readwrite("doc_len",       &DocResult::doc_len)
        .def_readwrite("disp_len",      &DocResult::disp_len)
        .def_readwrite("needle_offset", &DocResult::needle_offset)
        .def_readwrite("metadata",      &DocResult::metadata)
        .def_readwrite("text",          &DocResult::text);

    py::class_<Engine>(m, "Engine")
        .def(py::init<const std::vector<std::string>, const bool, const bool>(),
             "index_dirs"_a, "load_to_ram"_a, "get_metadata"_a)
        .def("find",
             &Engine::find,
             py::call_guard<py::gil_scoped_release>(),
             "query"_a,
             "Find the SA interval [lo, hi) per shard where query occurs.")
        .def("count",
             &Engine::count,
             py::call_guard<py::gil_scoped_release>(),
             "query"_a,
             "Count occurrences of query across the corpus.")
        .def("get_doc_by_rank",
             &Engine::get_doc_by_rank,
             py::call_guard<py::gil_scoped_release>(),
             "s"_a, "rank"_a, "needle_len"_a, "max_ctx_len"_a,
             "Retrieve the document at position rank in the SA of shard s.");
}
