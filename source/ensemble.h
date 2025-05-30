#ifndef _ENSEMBLE_H_
#define _ENSEMBLE_H_

#include <filesystem>
#include <vector>
#include <list>

#include "Model.h"
#include "state_merger.h"
#include "refinement.h"

void bagging(state_merger* merger, const std::string &output_file, int nr_estimators);

enum GenerationMode {
    Random,
    Greedy
};

enum VotingStrategy {
    Uniform,
    Weighted
};


class Ensemble {
    int size = 0;
    std::vector<Model> models;

    // Voting and weights computation
    VotingStrategy voting_strategy = Uniform;
    std::vector<double> weights;

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
     * Makes a prediction for the given trace. For now, uses a weighted average
     * vote to merge the predictions of single models.
     * @param trace trace to make a prediction for
     * @return the prediction
     */
    [[nodiscard]] double predict(trace* trace) const;

    [[nodiscard]] std::vector<double> compute_models_cross_entropy(int sample_size) const;

    void compute_weights(int sample_size);

    void set_equal_weights();

    friend class EnsembleFactory;
};


class EnsembleFactory {
    static refinement_list random(state_merger* merger);

    static refinement_list greedy(state_merger* merger);

    static refinement* get_random_ref(const refinement_set &s);

    static refinement_list refinements_by_mode(const GenerationMode &mode, state_merger* merger) {
        switch (mode) {
            case Random: return random(merger);
            case Greedy: return greedy(merger);
            default: throw std::invalid_argument("EnsembleFactory::refinements_by_mode: Invalid argument");
        }
    }

    static std::string modeToString(const GenerationMode mode) {
        switch (mode) {
            case Random: return "random";
            case Greedy: return "greedy";
            default: throw std::invalid_argument("EnsembleFactory::modeToString: Invalid argument");
        }
    }

    static std::string votingStrategyToString(const VotingStrategy strategy) {
        switch (strategy) {
            case Uniform: return "uniform";
            case Weighted: return "weighted";
            default: throw std::invalid_argument("EnsembleFactory::votingStrategyToString: Invalid argument");
        }
    }

    static GenerationMode stringToMode(const std::string &mode) {
        if (mode == "random") return Random;
        if (mode == "greedy") return Greedy;
        return Random;
    }

    static VotingStrategy stringToVotingStrategy(const std::string &strategy) {
        if (strategy == "uniform") return Uniform;
        if (strategy == "weighted") return Weighted;
        return Uniform;
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

public:
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
     * Loads and adds a single model to the ensemble.
     * @param ensemble the ensemble to add the model to
     * @param model_path path descriptor of the single model
     * @return number of added models (1 or 0)
     */
    static int add_single_model(Ensemble &ensemble, const std::string &model_path);

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
     */
    static void make_weighted(Ensemble &ensemble, const int sample_size) {
        ensemble.voting_strategy = Weighted;
        ensemble.compute_weights(sample_size);
    }

    /**
     * Writes the ensemble weights to a file.
     * @param ensemble
     * @param model_path
     */
    static void write_weights(const Ensemble &ensemble, const std::string &model_path);
};


#endif /* _ENSEMBLE_H_ */
