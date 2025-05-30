//
// Created by blaze on 14/05/2025.
//

#include "TestRunner.h"

#include <filesystem>

#include "csv.hpp"
#include "ensemble.h"
#include "input/parsers/csvparser.h"

std::optional<TestRunner<Ensemble> > TestRunner<Ensemble>::create_from_ensemble(
    const std::string &model_file,
    const int ensemble_size,
    const std::string &strategy_str
) {
    // Now, load the ensemble from files
    Ensemble ensemble;

    if (EnsembleFactory::add_model_collection(ensemble, model_file, ensemble_size) != ensemble_size) {
        std::cerr << "Failed to create ensemble from model: " << model_file << std::endl;
        return std::nullopt;
    }
    if (!EnsembleFactory::load_weights(ensemble, model_file, strategy_str)) {
        std::cerr << "Failed to load/create ensemble weights for model: " << model_file << std::endl;
        return std::nullopt;
    }

    // Setup output file stream
    std::ofstream output(model_file + ".ensemble.result");

    return std::make_optional<TestRunner>(ensemble, std::move(output));
}

std::optional<TestRunner<Model>> TestRunner<Model>::create_from_model(const std::string &model_file) {
    // Now, load the model from the file
    namespace fs = std::filesystem;

    // Check if file exists
    std::string filename = model_file + ".final.json";
    if (!fs::exists(filename) || !fs::is_regular_file(filename)) {
        std::cerr << "Model file does not exist: " << filename << std::endl;
        return std::nullopt;
    }
    // Try to open the file
    const auto file_stream = std::make_unique<std::ifstream>(filename);
    if (!file_stream->is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return std::nullopt;
    }
    // Create the model from the file contents
    std::cout << "Loading single model from file: " << filename << std::endl;
    const Model model = Model::from_apta_json(*file_stream);

    // Setup output file stream
    std::ofstream output(model_file + ".single.result");

    return std::make_optional<TestRunner>(model, std::move(output));
}

double compute_cross_entropy(const std::vector<double> &ps, const std::vector<double> &qs) {
    const size_t num_vals = ps.size();
    if (qs.size() != num_vals) {
        throw std::invalid_argument("Number of p values does not match number of q values");
    }

    // Compute normalization constant
    // constexpr double EPS = 1e-30; // Replace all ps smaller than EPS with EPS
    double p_sum = 0;
    double q_sum = 0;

    // The technique for dealing with 0s is to replace them with the average q value
    int q_num_zeros = 0;

    for (int i = 0; i < num_vals; i++) {
        p_sum += ps[i];
        q_sum += qs[i];
        if (qs[i] == 0.0) {
            q_num_zeros++;
        }
    }
    // Add the average q value times the number of zeros in qs to q_sum.
    // If qs are all zeros, then just set average to arbitrary value.
    const double q_avg = q_num_zeros != num_vals ? q_sum / static_cast<double>(num_vals - q_num_zeros) : 1.0;
    q_sum += q_avg * q_num_zeros;

    // Normalization constants are inverses of sums
    const double Np = 1 / p_sum;
    const double Nq = 1 / q_sum;

    // std::cout << "q_avg=" << q_avg << ", zeros=" << q_num_zeros << std::endl;

    // Compute cross-entropy
    double cross_entropy = 0;
    for (int i = 0; i < num_vals; i++) {
        cross_entropy -= ps[i] * log2(std::max(q_avg, qs[i]) * Nq);
        // std::printf("%d: p=%.6f q=%.6f, ce=%.6f\n", i, ps[i], qs[i], cross_entropy);
    }
    cross_entropy *= Np;

    return cross_entropy;
}

double compute_perplexity(const std::vector<double> &ps, const std::vector<double> &qs) {
    // Compute perplexity
    const double cross_entropy = compute_cross_entropy(ps, qs);
    const double perplexity = pow(2.0, cross_entropy);
    return perplexity;
}
