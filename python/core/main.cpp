#include <pybind11/pybind11.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/spdlog.h>

namespace py = pybind11;
using namespace pybind11::literals;

void init_blender(py::module&);
void init_clipping(py::module&);
void init_closest(py::module&);
void init_fairing(py::module&);
void init_fitting(py::module&);
void init_intersect(py::module&);
void init_io(py::module&);
void init_junction(py::module&);
void init_junction_features(py::module&);
void init_mapping(py::module&);
void init_render(py::module&);
void init_sketching(py::module&);
void init_stroke_graph(py::module&);
void init_region_cues(py::module& m);
void init_consolidation(py::module& m);
void init_modules(py::module& m) {
  init_blender(m);
  init_clipping(m);
  init_closest(m);
  init_fairing(m);
  init_fitting(m);
  init_intersect(m);
  init_io(m);
  init_junction(m);
  init_junction_features(m);
  init_mapping(m);
  init_render(m);
  init_sketching(m);
  init_stroke_graph(m);
  init_region_cues(m);
  init_consolidation(m);
}

namespace {

py::str build_string() {
#ifdef NDEBUG
  return "release";
#else
  return "debug";
#endif
}

template <typename Mutex>
class python_sink : public spdlog::sinks::base_sink<Mutex> {
protected:
  void sink_it_(const spdlog::details::log_msg& msg) override {
    spdlog::memory_buf_t formatted;
    spdlog::sinks::base_sink<Mutex>::formatter_->format(msg, formatted);
    py::print(fmt::to_string(formatted), "end"_a = "",
              "file"_a = py::module::import("sys").attr("stderr"));
  }

  void flush_() override {
    (void)py::module::import("sys").attr("stderr").attr("flush")();
  }
};
using python_sink_mt = python_sink<std::mutex>;
using python_sink_st = python_sink<spdlog::details::null_mutex>;

auto sink = std::make_shared<python_sink_mt>();
auto python_logger = std::make_shared<spdlog::logger>("python_logger", sink);

void set_log_level(const std::string& level) {
  spdlog::set_level(spdlog::level::from_str(level));
}

} // namespace

PYBIND11_MODULE(_sketching, m) {
#ifdef _WIN32
  // Pop up a dialog on failed assert so that we can attach a debugger at that point.
  // This is the default for Debug configurations, but not for RelWithDebInfo
  // configurations, even with /DNDEBUG off, it seems.
  _set_error_mode(_OUT_TO_MSGBOX);
#endif

  m.doc() = "Topology-aware sketching tools";

  m.def("build_string", &build_string);
  spdlog::set_default_logger(python_logger);
  spdlog::set_pattern("%^[%H:%M:%S.%e][%l]%$ %v (%s:%#)");
  m.def("set_log_level", &set_log_level);

  init_modules(m);
}
