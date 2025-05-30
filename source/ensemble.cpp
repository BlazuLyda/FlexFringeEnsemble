#include "ensemble.h"

#include <filesystem>
#include <ranges>
#include <fstream>
#include <random>

#include "refinement.h"
#include "greedy.h"
#include "Model.h"
#include "TestRunner.h"

/** todo: work in progress */

/** Ensemble methods **/
double Ensemble::predict(trace* trace) const {
    // std::cout << "Evaluating trace: " << trace->get_sequence() << std::endl;
    // Iterate through the models and evaluate the traces
    double prediction = 0;
    for (const auto &model: models) {
        const double prob = model.predict(trace);
        prediction += prob * weights[model.get_id()];
        // std::cout << "\tModel " << model.get_id() << ": w=" << weights[model.get_id()] << ", pred=" << prob << std::endl;
    }

    // Return weighted average with equal weights
    // std::cout << "\tWeighted average: " << prediction << std::endl;
    return prediction;
}

std::vector<double> Ensemble::compute_models_cross_entropy(const int sample_size) const {
    std::vector<ModelTrace> sample;
    sample.resize(sample_size);

    // Compute diff for each pair of models
    std::vector<double> diffs;
    diffs.reserve(models.size() * models.size());

    // Generate a sample of traces from each model
    for (const auto &model: models) {
        // Generate a sample
        for (int i = 0; i < sample_size; i++) {
            sample[i] = model.generate_trace();
        }

        // For all models compute a score on this set of traces
        for (const auto &other: models) {
            diffs.push_back(other.compute_diff(sample));
        }
    }

    // Now output the diffs separated by ';'
    std::cout << "diffs: ";
    for (const double &diff: diffs) {
        std::cout << diff << ";";
    }
    std::cout << std::endl;

    return diffs;
};

void Ensemble::set_equal_weights() {
    const double w = 1.0 / size;
    weights.resize(size, w);
}

void Ensemble::compute_weights(const int sample_size) {
    // If uniform, just set all weights to equal
    if (voting_strategy == Uniform) {
        set_equal_weights();
        return;
    }

    // Flat array representing the sample cross-entropy matrix, should have length n**2
    const int n = static_cast<int>(models.size());
    std::vector<double> H_flat = compute_models_cross_entropy(sample_size);
    assert(H_flat.size() == n * n);

    // Lambda function for 2d index to 1d index
    auto index1d = [n](const int row_nr, const int col_nr) constexpr {
        return row_nr * n + col_nr;
    };

    // 1st normalization step: take a difference of row baseline (values on diagonal) from the row
    std::vector<double> diagonal(n);
    for (int i = 0; i < n; i++) {
        diagonal[i] = H_flat[index1d(i, i)];
    }
    // Now subtract this from every row
    for (int i = 0; i < n; i++) {
        const double row_base = diagonal[i];
        for (int j = 0; j < n; j++) {
            H_flat[index1d(i, j)] -= row_base;
        }
    }

    // Now perform the real training
    std::vector<double> col_sums(n), row_sums(n);
    weights.resize(n, 1.0 / n);
    double weight_sum = 0;

    constexpr int EPOCHS = 10;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // Collect row/col sums
        for (int i = 0; i < n; i++) {
            col_sums[i] = 0;
            row_sums[i] = 0;
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                const int arr_index = index1d(i, j);
                const double w_term = weights[i] * weights[j];

                row_sums[i] += H_flat[arr_index] * w_term;
                col_sums[j] += H_flat[arr_index] * w_term;
            }
        }

        // Compute the ratios as row sum / column sum
        weight_sum = 0;
        for (int i = 0; i < n; i++) {
            if (col_sums[i] == 0.0) col_sums[i] = 1.0;
            weights[i] = row_sums[i] / col_sums[i];
            weight_sum += weights[i];
        }

        // Normalize the weights
        for (int i = 0; i < n; i++) {
            weights[i] = weights[i] / weight_sum;
        }
    }
}


/** Ensemble Factory methods **/
refinement_list greedy(state_merger* merger) {
    std::cout << "starting greedy merging" << std::endl;
    merger->get_eval()->initialize_after_adding_traces(merger);

    refinement_list all_refs{};

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

void bagging(state_merger* merger, const std::string &output_file, const int nr_estimators) {
    std::cout << "starting bagging" << std::endl;

    std::vector<Model> models = {};

    for (int i = 1; i <= nr_estimators; ++i) {
        auto all_refs = greedy(merger);

        // Create model for further evaluation using the merged apta
        Model new_model = Model::from_state_merger(merger);
        new_model.set_id(i);
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

    refinement_list all_refs{};

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

    refinement_list all_refs{};

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

refinement* EnsembleFactory::get_random_ref(const refinement_set &s) {
    if (s.empty()) return nullptr;

    static std::random_device rd;
    static std::mt19937 gen(rd() + std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution dist(0, static_cast<int>(s.size() - 1));

    const int index = dist(gen);
    const auto it = std::next(s.begin(), index);
    return *it;
}

Ensemble EnsembleFactory::generate(
    const std::string &mode_str,
    const std::string &strategy_str,
    state_merger* merger,
    const std::string &output_file,
    const int nr_estimators,
    const int sample_size
) {
    // Get the modes from the parameters
    const GenerationMode mode = stringToMode(mode_str);
    const VotingStrategy strategy = stringToVotingStrategy(strategy_str);
    std::cout << "Starting the creation of ensemble with mode: " << modeToString(mode) << std::endl;
    std::cout << "Selected voting strategy: " << votingStrategyToString(strategy) << std::endl;

    // Initialize the ensemble object
    Ensemble ensemble;
    ensemble.voting_strategy = strategy;
    ensemble.models.reserve(nr_estimators);
    std::cout << "Creating ensemble of size: " << nr_estimators << std::endl;

    for (int i = 1; i <= nr_estimators; ++i) {
        // Train next model
        auto all_refs = refinements_by_mode(mode, merger);

        // Save the model to a file
        merger->print_json(output_file + ".model." + std::to_string(i) + ".json");
        merger->print_dot(output_file + ".model." + std::to_string(i) + ".dot");
        std::cout << "Created model " << i << "/" << nr_estimators << std::endl;

        // Create the model object for further evaluation
        ensemble.add_model(std::move(Model::from_state_merger(merger)));

        // Undo the whole training process
        for (const auto &all_ref: std::ranges::reverse_view(all_refs)) {
            all_ref->undo(merger);
        }
        for (const auto &all_ref: all_refs) {
            all_ref->erase();
        }
    }

    // Compute the weights of the ensemble
    ensemble.compute_weights(sample_size);
    // Save the weights to a file
    write_weights(ensemble, output_file);

    std::cout << "Successfully created ensemble" << std::endl;
    return ensemble;
}

int EnsembleFactory::add_model_collection(Ensemble &ensemble, const std::string &model_path,
                                          const int collection_size) {
    // Iterate through model numbers and read the model files
    int i = 0;
    while (i != collection_size) {

        // Select the ith model
        std::string filename = model_path + ".model." + std::to_string(i + 1) + ".json";
        // Open the file
        auto input_maybe = open_file(filename);
        if (!input_maybe) {
            break;
        }
        // Create the model from the file contents
        std::cout << "Loading model " << ensemble.next_model_id() << " from file: " << filename << std::endl;
        Model model = Model::from_apta_json(input_maybe.value());
        ensemble.add_model(std::move(model));

        ++i;
    }
    std::cout << "Loaded ensemble of size " << i << " for training set: " << model_path << std::endl;
    return i;
}

int EnsembleFactory::add_single_model(Ensemble &ensemble, const std::string &model_path) {
    // Check if file exists
    const std::string filename = model_path + ".final.json";
    if (auto input_maybe = open_file(filename)) {
        // Create the model from the file contents
        std::cout << "Loading single model from file: " << filename << std::endl;

        Model model = Model::from_apta_json(input_maybe.value());
        ensemble.add_model(std::move(model));
        return 1;
    }
    return 0;
}

bool EnsembleFactory::load_weights(Ensemble &ensemble, const std::string &model_path, const std::string &strategy_str) {
    // Begin with setting the ensemble voting strategy
    ensemble.voting_strategy = stringToVotingStrategy(strategy_str);
    // std::cout << "Loading weights for strategy: " << strategy_str << " decoded to: " << votingStrategyToString(ensemble.voting_strategy) << std::endl;

    // If the voting strategy is Uniform, just set equal weights
    if (ensemble.voting_strategy == Uniform) {
        std::cout << "Setting uniform weights for model " << model_path << std::endl;
        ensemble.set_equal_weights();
        return true;
    }

    // Open the weights file
    const std::string filename = model_path + ".weights.txt";
    ListReader weights_reader;
    if (!weights_reader.init(filename)) {
        std::cerr << "Cannot initialize reader for the weights file: " << filename << std::endl;
        return false;
    }
    std::cout << "Reading weights from file " << filename << std::endl;

    // If there is a weights file, initialize the ensemble with it
    const size_t num_weights = weights_reader.get_size();

    // Check if the file contains the expected number of weights
    if (ensemble.size != num_weights) {
        std::cerr << "Weights file has mismatched number of entries (" << num_weights
                << ") for ensemble of size " << ensemble.size << std::endl;
        return false;
    }
    ensemble.weights.reserve(num_weights);

    // Read all the weights
    while (weights_reader.has_next()) {
        ensemble.weights.push_back(weights_reader.read_next());
    }
    return true;
}

void EnsembleFactory::write_weights(const Ensemble& ensemble, const std::string& model_path) {

    if (ensemble.voting_strategy == Uniform) {
        std::cout << "Skipping writing weights file for ensemble with uniform voting strategy." << std::endl;
        return;
    }

    // Open the file for writing the weights
    const std::string filename = model_path + ".weights.txt";
    std::cout << "Writing weights file for ensemble: " << filename << std::endl;
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    // First line: number of weights
    out << ensemble.weights.size() << std::endl;
    // One weight per line
    for (const double weight : ensemble.weights) {
        out << weight << std::endl;
    }
}
