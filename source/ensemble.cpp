#include <stdio.h>
#include <ranges>
#include <sstream>
#include <fstream>
#include <cstdlib>

#include "refinement.h"
#include "greedy.h"
#include "parameters.h"
#include "Model.h"

/** todo: work in progress */

refinement_list *greedy(state_merger *merger) {
    std::cerr << "starting greedy merging" << std::endl;
    merger->get_eval()->initialize_after_adding_traces(merger);

    auto *all_refs = new refinement_list();

    refinement *best_ref = merger->get_best_refinement();
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
};

void bagging(state_merger *merger, std::string output_file, int nr_estimators) {
    std::cerr << "starting bagging" << std::endl;

	std::vector<std::unique_ptr<Model>> models = {};

    for (int i = 0; i < nr_estimators; ++i) {
        refinement_list *all_refs = greedy(merger);

        // Save the created model
		auto new_model = Model::from_state_merger(merger);
		models.push_back(std::move(new_model));

		// Undo the whole learning process
        for (auto &all_ref: std::ranges::reverse_view(*all_refs)) {
            all_ref->undo(merger);
        }
        for (auto &all_ref: *all_refs) {
            all_ref->erase();
        }
        delete all_refs;
    }

    std::cerr << "ended bagging" << std::endl;

    // Write the model
    std::cerr << "writing model 1" << std::endl;
    std::ofstream output(output_file + "_model_1.dot");
    if (output.fail()) {
        throw std::ofstream::failure("Unable to open file for writing: " + output_file);
    }
    models.at(0)->write_dot(output);
};



