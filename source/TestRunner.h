//
// Created by blaze on 14/05/2025.
//

#ifndef TESTRUNNER_H
#define TESTRUNNER_H
#include <memory>
#include <string>

#include "csv.hpp"
#include "ensemble.h"
#include "Model.h"
#include "input/inputdatalocator.h"
#include "input/parsers/abbadingoparser.h"
#include "input/parsers/csvparser.h"

/**
 * Computes perplexity metric based on cross entropy between the real probabilities in ps and predicted
 * probabilities in qs.
 * @param ps real probabilities (can be unnormalized)
 * @param qs predicted probabilities (can be unnormalized)
 * @return the perplexity score
 */
double compute_perplexity(const std::vector<double> &ps, const std::vector<double> &qs);

double compute_cross_entropy(const std::vector<double> &ps, const std::vector<double> &qs);

template<typename P>
concept Predictor = requires(const P &p, trace* trace)
{
    { p.predict(trace) } -> std::convertible_to<double>;
};

template<Predictor T>
class TestRunner {
    // Predictor used to evaluate traces
    T predictor;

    // Rest of objects required for running tests
    std::unique_ptr<parser> test_parser;
    std::unique_ptr<reader_strategy> test_reader_strategy;

    // Test input / Results output
    std::ifstream input;
    std::ofstream output;

    // Perplexity
    bool compute_score = false;
    std::ifstream solutions;
    size_t sol_total_count = 0;
    size_t sol_read_count = 0;

    void init_test_reader(const std::string &test_file) {
        input = std::ifstream(test_file);

        // We stream the to predict traces into inputdata one by one to save memory
        // Set up the parser for the input stream
        if (INPUT_FILE.ends_with(".csv")) {
            test_parser = std::make_unique<csv_parser>(input, csv::CSVFormat().trim({' '}));
        } else {
            test_parser = std::make_unique<abbadingoparser>(input);
        }

        // Set up the reading strategy. Currently only sliding window and in-order traces are supported
        std::unique_ptr<reader_strategy> strategy;
        if (SLIDING_WINDOW) {
            test_reader_strategy = std::make_unique<slidingwindow>(SLIDING_WINDOW_SIZE, SLIDING_WINDOW_STRIDE,
                                                                   SLIDING_WINDOW_TYPE);
        } else {
            test_reader_strategy = std::make_unique<in_order>();
        }
    }

    void init_solution_reader(const std::string &solution_file) {
        solutions.open(solution_file);
        if (!solutions) {
            throw std::runtime_error("Failed to open file: " + solution_file);
        }
        if (!(solutions >> sol_total_count)) {
            throw std::runtime_error("Failed to read number of traces");
        }
    }

    double read_next_solution() {
        if (sol_read_count >= sol_total_count) {
            throw std::runtime_error("Exceeded the number of lines in the file");
        }
        if (double value; solutions >> value) {
            ++sol_read_count;
            return value;
        }
        throw std::runtime_error("Error reading value at index " + std::to_string(sol_read_count));
    }

public:
    TestRunner(const T &predictor, std::ofstream &&output)
        : predictor(predictor), output(std::move(output)) {
    }

    ~TestRunner() = default;

    static std::optional<TestRunner<Ensemble>> create_from_ensemble(const std::string &model_file, int ensemble_size);

    static std::optional<TestRunner<Model>> create_from_model(const std::string &model_file);



    void run(const std::string &test_file) {
        // Init the test file reader
        init_test_reader(test_file);

        inputdata idat = inputdata::with_alphabet_from(*inputdata_locator::get());

        std::optional<trace *> trace_maybe = idat.read_trace(*test_parser, *test_reader_strategy);

        // For computing perplexity
        std::vector<double> real_probs;
        std::vector<double> predicted_probs;

        while (trace_maybe) {
            const auto trace = *trace_maybe;
            const double prediction = predictor.predict(trace);

            // Write the prediction to the output
            output << prediction << std::endl;

            // Optionally compare against solution
            if (compute_score) {
                const double real = read_next_solution();
                real_probs.push_back(real);
                predicted_probs.push_back(prediction);
            }

            // TODO: Deleting the traces should probably also invalidate the trace pointers in inputdata,
            //  but since we have a separate inputdata local to this function it is sort of ok here?
            trace->erase();
            trace_maybe = idat.read_trace(*test_parser, *test_reader_strategy);
        }

        // Optionally compute perplexity
        if (compute_score) {
            const double perplexity = compute_perplexity(real_probs, predicted_probs);
            std::cout << "Final perplexity: " << perplexity << std::endl;
        }
    }

    /**
     * Evaluate all the traces in the given test set using the loaded model. Compute
     * Perplexity score on the predictions using the provided target solutions.
     * @param test_file name of file containing the test set
     * @param solution_file name of file containing target probabilities
     */
    void run(const std::string &test_file, const std::string &solution_file) {
        // Run the test file against the solutions
        compute_score = true;
        init_solution_reader(solution_file);
        run(test_file);
    }
};


#endif //TESTRUNNER_H
