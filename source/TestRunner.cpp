//
// Created by blaze on 14/05/2025.
//

#include "TestRunner.h"

#include <filesystem>

#include "csv.hpp"
#include "ensemble.h"
#include "input/parsers/csvparser.h"

std::optional<TestRunner<Ensemble>> TestRunner<Ensemble>::create_from_ensemble(const std::string &model_file) {
    // Now, load the ensemble from files
    const std::optional<Ensemble> ensemble = EnsembleFactory::load(model_file);
    if (!ensemble) return std::nullopt;

    // Setup output file stream
    std::ofstream output(model_file + ".ensemble.result");

    return std::make_optional<TestRunner<Ensemble>>(ensemble.value(), std::move(output));
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
    const Model model = Model::from_apta_json(1, *file_stream);

    // Setup output file stream
    std::ofstream output(model_file + ".single.result");

    return std::make_optional<TestRunner<Model>>(model, std::move(output));
}

