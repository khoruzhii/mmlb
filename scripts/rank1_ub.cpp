#include "ub.h"

#include "cnpy.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using fgs::Scheme;
using fgs::U16;
using fgs::U32;
using fgs::U64;
using Tensor = fgs::bco::Tensor;
using Clock = std::chrono::steady_clock;

constexpr U32 kFlips = 1'000'000'000;
constexpr int kThreads = 12;

struct Dataset {
    std::string stem;
    std::vector<Tensor> tensors;
    std::vector<U16> upper_bounds;
};

struct Job {
    std::size_t dataset;
    std::size_t row;
};

void add_rank_one(Tensor &tensor, U16 u, U16 v, U16 w) {
    for (; u != 0; u &= static_cast<U16>(u - 1)) {
        const int i = std::countr_zero(u);
        for (U16 y = v; y != 0; y &= static_cast<U16>(y - 1)) {
            const int j = std::countr_zero(y);
            for (U16 z = w; z != 0; z &= static_cast<U16>(z - 1)) {
                const int k = std::countr_zero(z);
                const int bit = (9 * i + j) * 9 + k;
                tensor[bit / 64] ^= U64{1} << (bit % 64);
            }
        }
    }
}

Tensor tensor_of(const std::vector<Scheme::Term> &terms) {
    Tensor tensor{};
    for (const Scheme::Term &term : terms) {
        if (term[0] != 0) {
            add_rank_one(tensor, term[0], term[1], term[2]);
        }
    }
    return tensor;
}

Dataset load_dataset(const std::string &stem) {
    const std::filesystem::path path =
        std::filesystem::path("data/rk1step") / ("t" + stem + ".npy");
    const cnpy::NpyArray array = cnpy::npy_load(path.string());
    if (array.shape.size() != 2 || array.shape[1] != 12 ||
        array.word_size != sizeof(U64) || array.fortran_order) {
        throw std::runtime_error(path.string() +
                                 " must be C-order uint64 with shape (N,12)");
    }

    Dataset result;
    result.stem = stem;
    result.tensors.resize(array.shape[0]);
    const U64 *data = array.data<U64>();
    for (std::size_t row = 0; row < result.tensors.size(); ++row) {
        std::copy_n(data + 12 * row, 12, result.tensors[row].begin());
        if ((result.tensors[row].back() & ~((U64{1} << 25) - 1)) != 0) {
            throw std::runtime_error(path.string() + " has nonzero padding");
        }
    }
    result.upper_bounds.resize(result.tensors.size());
    return result;
}

} // namespace

int main() {
    try {
        std::vector<Dataset> datasets;
        for (const char *stem : {"222", "223", "233", "333"}) {
            datasets.push_back(load_dataset(stem));
        }

        std::vector<Job> jobs;
        for (std::size_t dataset = 0; dataset < datasets.size(); ++dataset) {
            for (std::size_t row = 0;
                 row < datasets[dataset].tensors.size(); ++row) {
                jobs.push_back({dataset, row});
            }
        }

        std::atomic<std::size_t> next_job = 0;
        std::atomic<std::size_t> completed = 0;
        std::exception_ptr failure;
        std::mutex failure_mutex;
        const auto wall_start = Clock::now();

        std::vector<std::thread> workers;
        workers.reserve(kThreads);
        for (int thread = 0; thread < kThreads; ++thread) {
            workers.emplace_back([&] {
                try {
                    while (true) {
                        const std::size_t index = next_job.fetch_add(1);
                        if (index >= jobs.size()) {
                            break;
                        }
                        const Job job = jobs[index];
                        Dataset &dataset = datasets[job.dataset];
                        const Tensor &tensor = dataset.tensors[job.row];
                        const std::vector<Scheme::Term> terms =
                            fgs::ubd(tensor, kFlips);
                        if (tensor_of(terms) != tensor) {
                            throw std::runtime_error(
                                "upper-bound decomposition is incorrect");
                        }
                        dataset.upper_bounds[job.row] =
                            static_cast<U16>(terms.size());
                        const std::size_t done = completed.fetch_add(1) + 1;
                        if (done % 20 == 0 || done == jobs.size()) {
                            const double seconds = std::chrono::duration<double>(
                                                       Clock::now() - wall_start)
                                                       .count();
                            std::cout << "completed=" << done << '/'
                                      << jobs.size() << " wall_seconds="
                                      << std::fixed << std::setprecision(1)
                                      << seconds << '\n'
                                      << std::flush;
                        }
                    }
                } catch (...) {
                    std::lock_guard lock(failure_mutex);
                    if (!failure) {
                        failure = std::current_exception();
                    }
                    next_job = jobs.size();
                }
            });
        }
        for (std::thread &worker : workers) {
            worker.join();
        }
        if (failure) {
            std::rethrow_exception(failure);
        }
        const double wall_seconds =
            std::chrono::duration<double>(Clock::now() - wall_start).count();

        for (const Dataset &dataset : datasets) {
            const std::filesystem::path path =
                std::filesystem::path("data/rk1step") /
                ("d" + dataset.stem + ".npy");
            cnpy::npy_save(path.string(), dataset.upper_bounds.data(),
                           {dataset.upper_bounds.size()}, "w");

            std::map<U16, int> distribution;
            U32 sum = 0;
            for (const U16 rank : dataset.upper_bounds) {
                ++distribution[rank];
                sum += rank;
            }
            std::cout << 'd' << dataset.stem << ".npy shape=("
                      << dataset.upper_bounds.size() << ") dtype=uint16 mean="
                      << std::fixed << std::setprecision(6)
                      << static_cast<double>(sum) /
                             dataset.upper_bounds.size()
                      << " distribution:";
            for (const auto &[rank, count] : distribution) {
                std::cout << ' ' << rank << ':' << count;
            }
            std::cout << '\n';
        }
        std::cout << "flips=" << kFlips << " threads=" << kThreads
                  << " wall_seconds=" << std::fixed << std::setprecision(6)
                  << wall_seconds << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "rank1_ub: FAILED: " << error.what() << '\n';
        return 1;
    }
}
