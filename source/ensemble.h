#ifndef _ENSEMBLE_H_
#define _ENSEMBLE_H_

#include <vector>
#include <list>

#include "Model.h"
#include "state_merger.h"
#include "refinement.h"

void bagging(state_merger* merger, const std::string& output_file, int nr_estimators);

class Ensemble {
    std::vector<Model> models;

public:
    Ensemble() = default;

    ~Ensemble() = default;

    void add_model(Model&& model) {
        models.push_back(std::move(model));
    }

    /**
     * todo: refactor this to allow for different voting strategies
     *
     * Makes a prediction for the given trace. For now, uses a majority vote to merge the predictions
     * of single models.
     * @param trace trace to make a prediction for
     * @return the prediction
     */
    double predict(trace* trace) const;

    friend class EnsembleFactory;
};

enum GenerationMode {
    Random,
    Variety,
    Greedy,
};

class EnsembleFactory {
    static refinement_list random(state_merger* merger);
    static refinement_list greedy(state_merger* merger);

    static refinement* get_random_ref(const refinement_set &s);

    static refinement_list refinements_by_mode(const GenerationMode mode, state_merger* merger) {
        switch (mode) {
            case Random: return random(merger);
            case Greedy: return greedy(merger);
            case Variety: throw std::invalid_argument("EnsembleFactory::EnsembleFactory(): Not implemented");
            default: throw std::invalid_argument("EnsembleFactory::EnsembleFactory(): Invalid argument");
        }
    }

    static std::string modeToString(const GenerationMode mode) {
        switch (mode) {
            case Random: return "random";
            case Variety: return "variety";
            case Greedy: return "greedy";
            default: return "unknown";
        }
    }

    static GenerationMode stringToMode(const std::string &mode) {
        if (mode == "random") return Random;
        if (mode == "variety") return Variety;
        if (mode == "greedy") return Greedy;
        return Random;
    }

public:
    /**
     * Generates the ensemble of specified size by training the models on after another on the loaded data.
     * The training mode and number of models are specified by the parameters.
     * @param mode_str the training mode
     * @param merger the state merger object with loaded data and selected evaluation function
     * @param output_file where to save the learned models to
     * @param nr_estimators desired final size of the ensemble
     * @return pointer to the ensemble object that can be used for evaluation with the ensemble
     */
    static Ensemble generate(
        const std::string &mode_str,
        state_merger* merger,
        const std::string &output_file,
        int nr_estimators
    );

    /**
     * Loads an ensemble object from a collection of models.
     * @param model_path the location of the model files
     * @return pointer to ensemble object that can be used for evaluation
     */
    static std::optional<Ensemble> load(const std::string &model_path);
};


#endif /* _ENSEMBLE_H_ */
