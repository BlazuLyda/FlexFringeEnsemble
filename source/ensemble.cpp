#include "ensemble.h"

#include <filesystem>
#include <ranges>
#include <fstream>
#include <future>
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

std::vector<double> Ensemble::predict_all(const std::vector<trace*> &traces) const {

    // Predict asynchronously
    std::vector<std::future<std::vector<double>>> futures; // Vector of tasks which return a vector of predictions

    for (const auto &model: models) {
        futures.push_back(std::async(std::launch::async, [&model, &traces] {
            return model.predict_all(traces);
        }));
    }

    // Here aggregate the results
    std::vector<double> predictions(traces.size(), 0.0);
    for (size_t m = 0; m < models.size(); ++m) {
        const auto local_predictions = futures[m].get(); // waits for completion
        const double weight = weights.at(m);

        for (size_t i = 0; i < traces.size(); ++i) {
            predictions[i] += local_predictions[i] * weight;
        }
    }
    return predictions;
}

std::vector<double> Ensemble::compute_models_cross_entropy(const int sample_size) const {
    // Compute diff for each pair of models
    std::vector<double> diffs;
    diffs.reserve(models.size() * models.size());

    // Generate a sample of traces from each model
    for (const auto &model: models) {
        std::vector<ModelTrace> sample = model.generate_trace_set(sample_size);
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

void Ensemble::set_random_weights() {
    std::random_device rd;
    std::mt19937 gen(rd() ^ std::chrono::high_resolution_clock::now().time_since_epoch().count());
    // Seed with non-deterministic randomness
    std::exponential_distribution exp_dist(1.0);

    weights.resize(size);
    double sum = 0;

    for (int i = 0; i < size; ++i) {
        weights[i] = exp_dist(gen); // Sample exponential(1)
        sum += weights[i];
    }
    for (double &w: weights) {
        w /= sum; // Normalize to sum to 1
    }

    // Now output the weights separated by ';'
    std::cout << "weights: ";
    for (const double &w: weights) {
        std::cout << w << ";";
    }
    std::cout << std::endl;
}

void Ensemble::set_best_fit_weights(const int sample_size) {
    // Flat array representing the sample cross-entropy matrix, should have length n**2
    const int n = static_cast<int>(models.size());
    std::vector<double> H_flat = compute_models_cross_entropy(sample_size);
    assert(H_flat.size() == n * n);
    // Exit early if computing weights not needed
    if (voting_strategy != VoteStrat::Weighted) {
        return;
    }

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
    const GenMode mode = stringToMode(mode_str);
    const VoteStrat strategy = stringToVotingStrategy(strategy_str);
    std::cout << "Starting the creation of ensemble with mode: " << modeToString(mode) << std::endl;
    std::cout << "Selected voting strategy: " << votingStrategyToString(strategy) << std::endl;

    // Initialize the ensemble object
    Ensemble ensemble;
    ensemble.voting_strategy = strategy;
    ensemble.models.reserve(nr_estimators);
    std::cout << "Creating ensemble of size: " << nr_estimators << std::endl;

    for (int i = 0; i < nr_estimators; ++i) {

        // Mode number is the FIRST_ID of this session plus the number of already generated models
        const std::string model_number = std::to_string(FIRST_ID + i);
        const std::string json_file = output_file + ".model." + model_number + ".json";
        const std::string dot_file = output_file + ".model." + model_number + ".dot";

        // If the option to continue previous work is set, then check if model already exists
        if (CONTINUE_WORK) {
            if (auto input_maybe = open_file(json_file)) {
                ensemble.add_model(Model::from_apta_json(input_maybe.value()));
                std::cout << "Read already created model " << model_number << ": " << i+1 << "/" << nr_estimators << std::endl;
                continue;
            }
        }

        // Train next model
        auto all_refs = refinements_by_mode(mode, merger);

        // Save the model to a file
        merger->print_json(json_file);
        merger->print_dot(dot_file);
        std::cout << "Created model " << model_number << ": " << i+1 << "/" << nr_estimators << std::endl;

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

    // Compute and set the weights of the ensemble
    if (ensemble.voting_strategy == VoteStrat::Weighted) {
        ensemble.set_best_fit_weights(sample_size);
        write_weights(ensemble, output_file);
    }

    std::cout << "Successfully created ensemble" << std::endl;
    return ensemble;
}

int EnsembleFactory::add_model_collection(Ensemble &ensemble, const std::string &model_path,
                                          const int collection_size) {
    // Iterate through model numbers and read the model files
    int i = 0;
    while (i < collection_size) {
        // Select the ith model
        std::string filename = model_path + ".model." + std::to_string(i) + ".json";
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

int EnsembleFactory::add_selected_models(Ensemble &ensemble, const std::string &model_path, const std::string &models_str) {

    // Extract the model nums
    std::vector<std::string> model_nums;
    for (auto part : std::views::split(models_str, ';')) {
        model_nums.emplace_back(part.begin(), part.end());
    }

    // Iterate through model numbers and read the model files
    for (const std::string& model_num : model_nums) {
        // Select the ith model
        std::string full_model_path = model_path + ".model." + model_num + ".json";
        // Add the model from the file to the ensemble
        add_single_model(ensemble, full_model_path);
    }
    std::cout << "Loaded ensemble of size " << model_nums.size() << " for training set: " << model_path << std::endl;
    return static_cast<int>(model_nums.size());
}

int EnsembleFactory::add_single_model(Ensemble &ensemble, const std::string &full_model_path) {
    // Check if file exists
    if (auto input_maybe = open_file(full_model_path)) {
        // Create the model from the file contents
        std::cout << "Loading single model from file: " << full_model_path << std::endl;

        Model model = Model::from_apta_json(input_maybe.value());
        ensemble.add_model(std::move(model));
        return 1;
    }
    std::cout << "Could not load single model: " << full_model_path << std::endl;
    return 0;
}

bool EnsembleFactory::load_weights(Ensemble &ensemble, const std::string &model_path, const std::string &strategy_str) {

    // Begin with setting the ensemble voting strategy
    ensemble.voting_strategy = stringToVotingStrategy(strategy_str);
    // std::cout << "Loading weights for strategy: " << strategy_str << " decoded to: " << votingStrategyToString(ensemble.voting_strategy) << std::endl;

    switch (ensemble.voting_strategy) {
        case VoteStrat::Uniform:
            std::cout << "Setting uniform weights for model: " << model_path << std::endl;
            ensemble.set_equal_weights();
            return true;
        case VoteStrat::Random:
            std::cout << "Setting random weights for model: " << model_path << std::endl;
            ensemble.set_random_weights();
            return true;
        case VoteStrat::Weighted:
            return load_weights_from_file(ensemble, model_path);
        case VoteStrat::Precomputed:
            if (ENS_WEIGHTS.empty()) {
                std::cerr << "Must provide weights as params when using precomputed voting strategy" << std::endl;
                return false;
            }
            return load_weights_from_params(ensemble, ENS_WEIGHTS);
    }
    return false;
}

bool EnsembleFactory::load_weights_from_file(Ensemble &ensemble, const std::string &model_path) {
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

bool EnsembleFactory::load_weights_from_params(Ensemble &ensemble, const std::string &weight_str) {

    std::cout << "Reading weights from parameters" << std::endl;

    // Split the weights string into array of weight values
    std::vector<std::string> weights_vec;
    weights_vec.reserve(ensemble.size);
    for (auto part : std::views::split(weight_str, ';')) {
        weights_vec.emplace_back(part.begin(), part.end());
    }
    // Check if the number of weights matches number of models
    if (weights_vec.size() != ensemble.size) {
        std::cerr << "Weights parameter has mismatched number of values: " << weights_vec.size() << std::endl;
        return false;
    }

    // Parse the values
    ensemble.weights.reserve(ensemble.size);
    for (const std::string& str_value : weights_vec) {
        ensemble.weights.emplace_back(std::stoi(str_value));
    }
    return true;
}



void EnsembleFactory::write_weights(const Ensemble &ensemble, const std::string &model_path) {
    if (ensemble.voting_strategy != VoteStrat::Weighted) {
        std::cout << "Skipping writing weights file for ensemble with voting strategy: " << votingStrategyToString(ensemble.voting_strategy) << std::endl;
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
    for (const double weight: ensemble.weights) {
        out << weight << std::endl;
    }
}
