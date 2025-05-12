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
int Ensemble::predict(trace* trace) const {
    std::cout << "Evaluating trace: " << trace->get_sequence() << std::endl;
    // Assume the prediction can only have two outcomes: 0 or 1
    // Iterate through the models and evaluate the traces
    int votes_zero = 0;
    int votes_one = 0;
    for (const auto& model : models) {
        const double prediction = model->evaluate(trace);
        std::cout << "\tModel " << model->get_id() << " predicted: " << prediction << std::endl;
        // In majority voting, don't pay attention to value of the probability
        if (prediction > 0) {
            votes_one++;
        } else {
            votes_zero++;
        }
    }

    // Return majority vote
    std::cout << "\tVotes for 1: " << votes_one << ", votes for 0: " << votes_zero << std::endl;
    return votes_one > votes_zero;
}

/** Ensemble Factory methods **/
std::unique_ptr<refinement_list> greedy(state_merger* merger) {
    std::cout << "starting greedy merging" << std::endl;
    merger->get_eval()->initialize_after_adding_traces(merger);

    auto all_refs = std::make_unique<refinement_list>();

    refinement* best_ref = merger->get_best_refinement();
    while (best_ref != nullptr) {
        std::cout << " ";
        best_ref->print_short();
        std::cout << " ";
        std::cout.flush();

        best_ref->doref(merger);
        all_refs->push_back(best_ref);
        best_ref = merger->get_best_refinement();
    }
    std::cout << "no more possible merges" << std::endl;
    return all_refs;
}

void bagging(state_merger* merger, std::string output_file, const int nr_estimators) {
    std::cout << "starting bagging" << std::endl;

    std::vector<std::unique_ptr<Model> > models = {};

    for (int i = 1; i <= nr_estimators; ++i) {
        auto all_refs = greedy(merger);

        // Create model for further evaluation using the merged apta
        auto new_model = Model::from_state_merger(i, merger);
        models.push_back(std::move(new_model));

        // Save the merged apta to a file
        merger->print_json(output_file + ".model." + std::to_string(i) + ".json");

        // Undo the whole learning process
        for (const auto &all_ref: std::ranges::reverse_view(*all_refs)) {
            all_ref->undo(merger);
        }
        for (const auto &all_ref: *all_refs) {
            all_ref->erase();
        }
    }

    std::cout << "ended bagging" << std::endl;

    // Write the model
    std::cout << "writing model 1" << std::endl;
    std::ofstream output(output_file + "_model_1.dot");
    if (output.fail()) {
        throw std::ofstream::failure("Unable to open file for writing: " + output_file);
    }
    models.at(0)->write_dot(output);
}


/**
 * Creates a model using merge selection process that always selects a random merge from the list of available
 * consistent merges.
 * @param merger state merger used to peform the merging process
 * @return list of refinements that lead to the creation of the model
 */
std::unique_ptr<refinement_list> EnsembleFactory::random(state_merger* merger) {
    std::cout << "starting random merging" << std::endl;
    merger->get_eval()->initialize_after_adding_traces(merger);

    auto all_refs = std::make_unique<refinement_list>();

    const refinement_set* possible_refs = merger->get_possible_refinements();
    while (!possible_refs->empty()) {

        // Select a uniformly random refinement from the set of possible refinements
        refinement* random_ref = get_random_ref(*possible_refs);
        std::cout << " ";
        random_ref->print_short();
        std::cout << " ";
        std::cout.flush();

        random_ref->doref(merger);
        all_refs->push_back(random_ref);
        possible_refs = merger->get_possible_refinements();
    }
    std::cout << "no more possible merges" << std::endl;
    return all_refs;

};

refinement* EnsembleFactory::get_random_ref(const refinement_set& s) {
    if (s.empty()) return nullptr;

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution dist(0, static_cast<int>(s.size() - 1));

    const int index = dist(gen);
    const auto it = std::next(s.begin(), index);
    return *it;
}

std::unique_ptr<Ensemble> EnsembleFactory::generate(
    const GenerationMode mode,
    state_merger* merger,
    const std::string& output_file,
    const int nr_estimators
) {
    std::cout << "Starting the creation of ensemble with mode: " << modeToString(mode) << std::endl;
    // Initialize the ensemble object
    auto ensemble = std::make_unique<Ensemble>();

    for (int i = 1; i <= nr_estimators; ++i) {
        // Train next model
        auto all_refs = refinements_by_mode(mode, merger);

        // Create the model object for further evaluation
        auto new_model = Model::from_state_merger(i, merger);
        ensemble->add_model(std::move(new_model));

        // Save the model to a file
        merger->print_json(output_file + ".model." + std::to_string(i) + ".json");
        merger->print_dot(output_file + ".model." + std::to_string(i) + ".dot");
        std::cout << "Created model " << i << "/" << nr_estimators << std::endl;

        // Undo the whole training process
        for (const auto &all_ref: std::ranges::reverse_view(*all_refs)) {
            all_ref->undo(merger);
        }
        for (const auto &all_ref: *all_refs) {
            all_ref->erase();
        }
    }

    std::cout << "Ended the creation of ensemble" << std::endl;
    return ensemble;
}

std::unique_ptr<Ensemble> EnsembleFactory::load(const std::string &model_path) {
    // Initialize the ensemble object
    auto ensemble = std::make_unique<Ensemble>();

    // Iterate through model numbers and read the model files
    int i = 1;
    while (true) {
        // Select the ith model
        std::string filename = model_path + ".model." + std::to_string(i) + ".json";
        namespace fs = std::filesystem;

        // If json file of the model exists, then read it and add to ensemble
        if (fs::exists(filename) && fs::is_regular_file(filename)) {
            auto file_stream = std::make_unique<std::ifstream>(filename);
            if (file_stream->is_open()) {
                std::cout << "Loading model " << i << " from file: " << filename << std::endl;
                auto model = Model::from_apta_json(i, *file_stream);
                ensemble->add_model(std::move(model));
            } else {
                throw std::runtime_error("Unable to open file: " + filename);
            }
        } else {
            // No more model files to read
            break;
        }
        ++i;
    }

    std::cout << "Loaded ensemble of size " << ensemble->models.size() << " for training set: " << model_path << std::endl;
    return ensemble;
};
