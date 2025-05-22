#include "ensemble.h"

#include <filesystem>
#include <ranges>
#include <fstream>
#include <random>

#include "refinement.h"
#include "greedy.h"
#include "Model.h"

/** todo: work in progress */

/** Ensemble methods **/
double Ensemble::predict(trace* trace) const {
    std::cout << "Evaluating trace: " << trace->get_sequence() << std::endl;
    // Iterate through the models and evaluate the traces
    double sum_of_probs = 0;
    for (const auto& model : models) {
        const double prob = model.predict(trace);
        sum_of_probs += prob;
        std::cout << "\tModel " << model.get_id() << " predicted Pr = " << prob << std::endl;
    }

    // Return weighted average with equal weights
    const double weighted_avg = sum_of_probs / static_cast<double>(models.size());
    std::cout << "\tWeighted average: " << weighted_avg << std::endl;
    return weighted_avg;
}

/** Ensemble Factory methods **/
refinement_list greedy(state_merger* merger) {
    std::cout << "starting greedy merging" << std::endl;
    merger->get_eval()->initialize_after_adding_traces(merger);

    refinement_list all_refs {};

    refinement* best_ref = merger->get_best_refinement();
    while (best_ref != nullptr) {
        std::cout << " ";
        best_ref->print_short();
        std::cout << " ";
        std::cout.flush();

        best_ref->doref(merger);
        all_refs.push_back(best_ref);
        best_ref = merger->get_best_refinement();
    }
    std::cout << "no more possible merges" << std::endl;
    return all_refs;
}

void bagging(state_merger* merger, const std::string& output_file, const int nr_estimators) {
    std::cout << "starting bagging" << std::endl;

    std::vector<Model> models = {};

    for (int i = 1; i <= nr_estimators; ++i) {
        auto all_refs = greedy(merger);

        // Create model for further evaluation using the merged apta
        Model new_model = Model::from_state_merger(i, merger);
        models.push_back(new_model);

        // Save the merged apta to a file
        merger->print_json(output_file + ".model." + std::to_string(i) + ".json");

        // Undo the whole learning process
        for (const auto &all_ref: std::ranges::reverse_view(all_refs)) {
            all_ref->undo(merger);
        }
        for (const auto &all_ref: all_refs) {
            all_ref->erase();
        }
    }

    std::cout << "ended bagging" << std::endl;
}


/**
 * Creates a model using merge selection process that always selects a random merge from the list of available
 * consistent merges.
 * @param merger state merger used to perform the merging process
 * @return list of refinements that lead to the creation of the model
 */
refinement_list EnsembleFactory::random(state_merger* merger) {
    std::cout << "starting random merging" << std::endl;
    merger->get_eval()->initialize_after_adding_traces(merger);

    refinement_list all_refs {};

    const refinement_set* possible_refs = merger->get_possible_refinements();
    while (!possible_refs->empty()) {

        // Select a uniformly random refinement from the set of possible refinements
        refinement* random_ref = get_random_ref(*possible_refs);
        std::cout << " ";
        random_ref->print_short();
        std::cout << " ";
        std::cout.flush();

        random_ref->doref(merger);
        all_refs.push_back(random_ref);
        possible_refs = merger->get_possible_refinements();
    }
    std::cout << "no more possible merges" << std::endl;
    return all_refs;
}

/**
 * Creates a model using greedy merge selection process, which always picks the best scored merge. This is intended
 * to be used with parameters such as "--random", which multiplies the merge scores by a random number to provide
 * some controlled nondeterminism.
 * @param merger state merger used to perform the merging process
 * @return list of refinements that lead to the creation of the model
 */
refinement_list EnsembleFactory::greedy(state_merger* merger) {
    std::cout << "starting random merging" << std::endl;
    merger->get_eval()->initialize_after_adding_traces(merger);

    refinement_list all_refs {};

    refinement* best_ref = merger->get_best_refinement();
    while (best_ref != nullptr) {

        std::cout << " ";
        best_ref->print_short();
        std::cout << " ";
        std::cout.flush();

        best_ref->doref(merger);
        all_refs.push_back(best_ref);
        best_ref = merger->get_best_refinement();
    }
    std::cout << "no more possible merges" << std::endl;
    return all_refs;
};

refinement* EnsembleFactory::get_random_ref(const refinement_set& s) {
    if (s.empty()) return nullptr;

    static std::random_device rd;
    static std::mt19937 gen(rd() + std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution dist(0, static_cast<int>(s.size() - 1));

    const int index = dist(gen);
    const auto it = std::next(s.begin(), index);
    return *it;
}

Ensemble EnsembleFactory::generate(
    const std::string& mode_str,
    state_merger* merger,
    const std::string& output_file,
    const int nr_estimators
) {
    // Get the mode from the string
    const GenerationMode mode = stringToMode(mode_str);

    std::cout << "Starting the creation of ensemble with mode: " << modeToString(mode) << std::endl;
    // Initialize the ensemble object
    Ensemble ensemble;

    for (int i = 1; i <= nr_estimators; ++i) {
        // Train next model
        auto all_refs = refinements_by_mode(mode, merger);

        // Save the model to a file
        merger->print_json(output_file + ".model." + std::to_string(i) + ".json");
        merger->print_dot(output_file + ".model." + std::to_string(i) + ".dot");
        std::cout << "Created model " << i << "/" << nr_estimators << std::endl;

        // Create the model object for further evaluation
        ensemble.add_model(std::move(Model::from_state_merger(i, merger)));

        // Undo the whole training process
        for (const auto &all_ref: std::ranges::reverse_view(all_refs)) {
            all_ref->undo(merger);
        }
        for (const auto &all_ref: all_refs) {
            all_ref->erase();
        }
    }

    std::cout << "Ended the creation of ensemble" << std::endl;
    return ensemble;
}

std::optional<Ensemble> EnsembleFactory::load(const std::string &model_path) {
    // Initialize the ensemble object
    Ensemble ensemble;

    // Iterate through model numbers and read the model files
    int i = 1;
    while (true) {
        // Select the ith model
        std::string filename = model_path + ".model." + std::to_string(i) + ".json";
        namespace fs = std::filesystem;

        // Check if the model file exists, if not end reading models
        if (!fs::exists(filename) || !fs::is_regular_file(filename)) {
            break;
        }
        // Try to open the file
        const auto file_stream = std::make_unique<std::ifstream>(filename);
        if (!file_stream->is_open()) {
            std::cerr << "Unable to open file: " << filename << std::endl;
            return std::nullopt;
        }
        // Create the model from the file contents
        std::cout << "Loading model " << i << " from file: " << filename << std::endl;
        Model model = Model::from_apta_json(i, *file_stream);
        ensemble.add_model(std::move(model));

        ++i;
    }

    std::cout << "Loaded ensemble of size " << ensemble.models.size() << " for training set: " << model_path << std::endl;
    return ensemble;
};
