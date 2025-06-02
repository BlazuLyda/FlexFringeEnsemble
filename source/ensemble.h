#ifndef _ENSEMBLE_H_
#define _ENSEMBLE_H_

#include <filesystem>
#include <vector>
#include <list>

#include "Model.h"
#include "state_merger.h"
#include "refinement.h"

void bagging(state_merger* merger, const std::string &output_file, int nr_estimators);

enum class GenMode {
    Random,
    Greedy
};

enum class VoteStrat {
    Uniform,
    Weighted,
    Random,
    Precomputed
};


class Ensemble {
    int size = 0;
    std::vector<Model> models;

    // Voting and weights computation
    VoteStrat voting_strategy = VoteStrat::Uniform;
    std::vector<double> weights;

    // Weight generation functions for different strategies
    void set_equal_weights();
    void set_random_weights();
    void set_best_fit_weights(int sample_size);

public:
    Ensemble() = default;

    ~Ensemble() = default;

    void add_model(Model &&model) {
        model.set_id(size);
        size++;
        models.push_back(std::move(model));
    }

    [[nodiscard]] int next_model_id() const {
        return size;
    }

    /**
     * Computes the ensemble prediction for the given trace. Internally, the trace is evaluated on
     * each of the ensemble models and then the predictions are combined using a weighted average
     * with weights defined by the ensemble voting strategy.
     * @param trace trace to make a prediction for
     * @return the prediction value between 0 and 1 inclusive
     */
    [[nodiscard]] double predict(trace* trace) const;

    /**
     * Computes pairwise sample cross entropy score between the models of the ensemble. For each model
     * it creates a kind of sample set of size sample_size. The set consists of unique traces generated
     * by random walks in the model. Each trace also contains the target probability. Then for each of
     * these sample sets, the cross entropy metric is evaluated.
     *
     * The output is a flattened 2d matrix of dimensions n x n, where n is the number of models in the
     * ensemble. Each row of the matrix (n consecutive values) contains the cross entropy metric evaluated
     * on the same sample set.
     * @param sample_size the size of the sample used to compute the cross entropy score
     * @return flattened 2d array containing the pairwise cross entropy scores
     */
    [[nodiscard]] std::vector<double> compute_models_cross_entropy(int sample_size) const;

    friend class EnsembleFactory;
};


class EnsembleFactory {
    static refinement_list random(state_merger* merger);

    static refinement_list greedy(state_merger* merger);

    static refinement* get_random_ref(const refinement_set &s);

    static refinement_list refinements_by_mode(const GenMode &mode, state_merger* merger) {
        switch (mode) {
            case GenMode::Random: return random(merger);
            case GenMode::Greedy: return greedy(merger);
            default: throw std::invalid_argument("EnsembleFactory::refinements_by_mode: Invalid argument");
        }
    }

    static std::string modeToString(const GenMode mode) {
        switch (mode) {
            case GenMode::Random: return "random";
            case GenMode::Greedy: return "greedy";
            default: throw std::invalid_argument("EnsembleFactory::modeToString: Invalid argument");
        }
    }

    static GenMode stringToMode(const std::string &mode) {
        if (mode == "random") return GenMode::Random;
        if (mode == "greedy") return GenMode::Greedy;
        return GenMode::Random;
    }

    static std::string votingStrategyToString(const VoteStrat strategy) {
        switch (strategy) {
            case VoteStrat::Uniform: return "uniform";
            case VoteStrat::Weighted: return "weighted";
            case VoteStrat::Random: return "random";
            case VoteStrat::Precomputed: return "precomputed";
            default: throw std::invalid_argument("EnsembleFactory::votingStrategyToString: Invalid argument");
        }
    }


    static std::optional<std::ifstream> open_file(const std::string &filename) {
        namespace fs = std::filesystem;
        // Check if file exists
        if (!fs::exists(filename) || !fs::is_regular_file(filename)) {
            std::cerr << "File does not exist: " << filename << std::endl;
            return std::nullopt;
        }
        // Try to open the file
        std::ifstream file_stream(filename);
        if (!file_stream.is_open()) {
            std::cerr << "Unable to open file: " << filename << std::endl;
            return std::nullopt;
        }
        return file_stream;
    }

    static bool load_weights_from_file(Ensemble &ensemble, const std::string &model_path);

    static bool load_weights_from_params(Ensemble &ensemble, const std::string &weight_str);

public:

    static VoteStrat stringToVotingStrategy(const std::string &strategy) {
        if (strategy == "uniform") return VoteStrat::Uniform;
        if (strategy == "weighted") return VoteStrat::Weighted;
        if (strategy == "random") return VoteStrat::Random;
        if (strategy == "precomputed") return VoteStrat::Precomputed;
        return VoteStrat::Uniform;
    }

    /**
     * Generates the ensemble of specified size by training the models on after another on the loaded data.
     * The training mode and number of models are specified by the parameters.
     * @param mode_str the training mode
     * @param strategy_str the voting strategy
     * @param merger the state merger object with loaded data and selected evaluation function
     * @param output_file where to save the learned models to
     * @param nr_estimators desired final size of the ensemble
     * @param sample_size sample size for the weight computation with voting strategy Weighted
     * @return pointer to the ensemble object that can be used for evaluation with the ensemble
     */
    static Ensemble generate(
        const std::string &mode_str,
        const std::string &strategy_str,
        state_merger* merger,
        const std::string &output_file,
        int nr_estimators,
        int sample_size
    );

    /**
     * Loads and adds a collection of models to the ensemble.
     * @param ensemble the ensemble to add the models to
     * @param model_path path descriptor of the models
     * @param collection_size
     * @return number of added models
     */
    static int add_model_collection(Ensemble &ensemble, const std::string &model_path, int collection_size);

    /**
     * Loads and adds models with specified ids from a bigger collection.
     * @param ensemble the ensemble to add the models to
     * @param model_path path descriptor of the model collection
     * @param models_str string containing the numbers of models to load, delimited by ';'
     * @return number of added models
     */
    static int add_selected_models(Ensemble &ensemble, const std::string &model_path, const std::string &models_str);

    /**
     * Loads and adds a single model to the ensemble.
     * @param ensemble the ensemble to add the model to
     * @param full_model_path full filename of the model to load (with suffix)
     * @return number of added models (1 or 0)
     */
    static int add_single_model(Ensemble &ensemble, const std::string &full_model_path);

    /**
     * Loads and adds weights from a file to the ensemble. The ensemble voting strategy must be set
     * before invoking this method.
     * @param ensemble the ensemble to add weights to
     * @param model_path identifier of the weights
     * @param strategy_str voting strategy describing the type of weights
     * @return true if weights were successfully read from file
     */
    static bool load_weights(Ensemble &ensemble, const std::string &model_path, const std::string &strategy_str);

    /**
     * Sets the voting strategy Weighted on the ensemble and computes the weights for the ensemble.
     * @param ensemble the ensemble to make weighted
     * @param sample_size sample size used for computing the inter model cross entropy
     * @param strategy_str strategy of the ensemble
     */
    static void compute_diffs(Ensemble &ensemble, const int sample_size, const std::string &strategy_str) {
        ensemble.voting_strategy = stringToVotingStrategy(strategy_str);
        ensemble.set_best_fit_weights(sample_size);
    }

    /**
     * Writes the ensemble weights to a file.
     * @param ensemble
     * @param model_path
     */
    static void write_weights(const Ensemble &ensemble, const std::string &model_path);
};


#endif /* _ENSEMBLE_H_ */
