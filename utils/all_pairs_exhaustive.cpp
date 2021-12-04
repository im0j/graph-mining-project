#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

#include "Eigen/Dense"
#include "cnpy.h"

using item_t = std::tuple<float, long, long>;

bool cmp_item(const item_t& item1, const item_t& item2)
{
    // Since we have to store k-largest items, heap should be a min-heap, so that
    // smallest item in the heap is popped efficiently.
    const float sim1 = std::get<0>(item1);
    const float sim2 = std::get<0>(item2);
    return sim1 > sim2;
}

int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cerr << "usage: " << argv[0] << "input k output" << std::endl;
        std::cerr << "example: " << argv[0] << " h.npy 1000 same_node.csv" << std::endl;
        return EXIT_SUCCESS;
    }

    unsigned long k = std::strtol(argv[2], NULL, 10);

    std::cout << "Loaded matrix file...";
    cnpy::NpyArray h_npy_arr = cnpy::npy_load(argv[1]);
    Eigen::MatrixXf h = Eigen::Map<Eigen::MatrixXf>(h_npy_arr.data<float>(), h_npy_arr.shape[0], h_npy_arr.shape[1]);
    std::cout << "Done.\n";
    std::cout << "Precomputing norms...";
    Eigen::VectorXf norms = h.rowwise().norm();
    std::cout << "Done.\n";

    std::vector<item_t> top_k;
    std::make_heap(top_k.begin(), top_k.end(), cmp_item);

    long num_nodes = h.rows();
    // long num_nodes = 1000;
    auto t0 = std::chrono::system_clock::now();
    for (long node1 = 0; node1 < num_nodes; node1++) {
        if (node1 % 10 == 0) {
            auto t1 = std::chrono::system_clock::now();
            auto td = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e9;
            t0 = t1;
            std::cout << "Progress: " << std::setw(5) << node1 << " / " << num_nodes;
            std::cout << " (took " << td << " s)" << std::endl;
        }
        auto v1 = h.row(node1);
        auto norm1 = norms.row(node1);
        for (long node2 = node1 + 1; node2 < num_nodes; node2++) {
            auto v2 = h.row(node2);
            auto norm2 = norms.row(node2);
            auto sim = v1.dot(v2) / (norm1 * norm2);

            top_k.emplace_back(sim, node1, node2);
            std::push_heap(top_k.begin(), top_k.end(), cmp_item);
            if (top_k.size() > k) {
                std::pop_heap(top_k.begin(), top_k.end(), cmp_item);
                top_k.pop_back();
            }
        }
    }

    std::sort_heap(top_k.begin(), top_k.end(), cmp_item);

    std::string csv_fname = argv[3];
    std::ofstream csv_f(csv_fname);
    if (!csv_f.is_open()) {
        std::cerr << "Failed to open " << csv_fname << "\n";
        return EXIT_FAILURE;
    }
    csv_f << "sim,source,target\n";
    for (auto item : top_k) {
        csv_f << std::get<0>(item) << "," << std::get<1>(item) << "," << std::get<2>(item) << "\n";
    }

    return EXIT_SUCCESS;
}
