//
// Created by blaze on 07.05.25.
//

#include "Model.h"

#include <random>
#include <ranges>

#include "alergia.h"
#include "TestRunner.h"
#include "input/inputdatalocator.h"
#include "utility/loguru.hpp"


double Model::predict(trace* trace) const {

    const ModelNode* node = &get_node(root_number);
    auto trace_it = trace->get_head();
    double prob = 1;

    // std::cout << "Starting evaluation on model " << id << std::endl;
    // Iterate through the trace while keeping the position in the graph
    while (trace_it != nullptr && !trace_it->is_final()) {
        // Get the transition corresponding to the trace symbol
        auto transition_maybe = node->follow(trace_it->get_symbol());
        // If trace follows a non-existing transition
        if (!transition_maybe) {
            // std::cout << "Followed a non-existing path: " << trace_it->get_symbol() << std::endl;
            return 0;
        }
        const ModelEdge& transition = transition_maybe.value().get();
        // Update the probability
        prob *= static_cast<double>(transition.count) / node->size;
        // Move to the target of the transition edge
        node = &get_node(transition.get_target());
        // std::cout << "Followed to node: " << node_it->number << ", with path: " << trace_it->get_symbol() << std::endl;
        trace_it = trace_it->future();
    }

    // Now check if the trace ends in an accepting state
    // std::cout << "Finished in node: " << node_it->number << " with final count: " << node_it->final << std::endl;
    if (node->final == 0) return 0;
    prob *= static_cast<double>(node->final) / node->size;
    return prob;
}

double Model::predict(const ModelTrace &trace) const {

    const ModelNode* node = &get_node(root_number);
    double prob = 1;

    for (const int symbol : trace.symbols) {
        // Get the transition corresponding to the trace symbol
        auto transition_maybe = node->follow(symbol);
        // If trace follows a non-existing transition
        if (!transition_maybe) {
            return 0;
        }
        const ModelEdge& transition = transition_maybe.value().get();
        // Update the probability
        prob *= static_cast<double>(transition.count) / node->size;
        // Move to the target of the transition edge
        node = &get_node(transition.get_target());
    }

    // Now check if the trace ends in an accepting state
    if (node->final == 0) return 0;
    prob *= static_cast<double>(node->final) / node->size;
    if (prob < 0 || prob > 1) {
        std::cerr << "invalid prob: " << prob << std::endl;
        throw std::invalid_argument("prob must be between 0 and 1");
    }
    return prob;
}


ModelTrace Model::generate_trace() const {

    // Initialize variables
    ModelTrace trace;
    std::random_device rd;
    std::mt19937 gen(rd());
    const ModelNode* current = &get_node(root_number);

    // Do a random walk
    while (true) {

        // Select a random int from 0 to total node count
        std::uniform_int_distribution dist(0, current->size - 1);
        const int choice = dist(gen);

        // Check if trace should finish in the node
        int total = current->final;
        if (choice < total) {
            trace.prob *= static_cast<double>(current->final) / current->size;
            return trace;
        }

        // Check which transition should be followed
        for (const auto &transition: current->edges | std::views::values) {
            total += transition.count;
            if (choice < total) {
                // Current transition selected
                trace.symbols.push_back(transition.symbol);
                trace.prob *= static_cast<double>(transition.count) / current->size;
                current = &get_node(transition.get_target());
                break;
            }
        }
    }
}

double Model::compute_diff(const std::vector<ModelTrace> &traces) const {

    std::vector<double> real;
    std::vector<double> predicted;
    real.reserve(traces.size());
    predicted.reserve(traces.size());

    // Collect real and predicted probabilities
    for (const auto &trace: traces) {
        real.push_back(trace.prob);
        const double prediction = predict(trace);
        predicted.push_back(prediction);
    }
    return compute_cross_entropy(real, predicted);
}

Model Model::from_state_merger(const int id, state_merger* merger) {
    // Get the apta from the merger
    apta* apta = merger->get_aut();
    apta_node* root = apta->get_root();

    // Create a new model
    Model model(id);

    // Copy all the apta nodes
    for (auto ait = merged_APTA_iterator(root); *ait != nullptr; ++ait) {
        // Create a new node and transfer ownership to the model
        apta_node& apta_node = **ait;
        model.add_node(ModelNode::from_apta_node(apta_node));
    }
    // Set the root number (this is always -1?)
    model.root_number = root->get_number();

    // Copy all the apta transitions
    for (auto ait = merged_APTA_iterator(root); *ait != nullptr; ++ait) {
        // Get the parent node
        apta_node& node = **ait;
        ModelNode& model_node = model.nodes.at(node.get_number());
        // Add the edges
        model_node.add_edges_from_apta(node);
    }
    return model;
}

ModelNode ModelNode::from_apta_node(apta_node& node) {
    return {
        node.get_number(),
        node.get_size(), // TODO: check if these are correct, or should I always call find()
        node.get_final()
    };
}

void ModelNode::add_edges_from_apta(apta_node& node) {
    // Assume the model is trained with Alergia data
    auto* node_data = dynamic_cast<alergia_data *>(node.get_data());

    // Iterate through the edges and add them
    for (auto it = node.guards_start(); it != node.guards_end(); ++it) {
        // Get edge data
        const int label = it->first;
        const int edge_count = node_data->count(it->first);

        // There is some very weird behaviour where some edges exist, but they have count 0 and
        // their target is nullptr. TODO: Investigate this
        if (edge_count == 0) {
            continue;
        }

        apta_node& child = *it->second->get_target()->find();

        // Construct and add the edge
        add_edge(ModelEdge(label, edge_count, child.get_number()));
    }
}


void Model::write_dot(std::ostream &out) const {
    if (!out) {
        LOG_S(ERROR) << "The output is not specified\n";
        return;
    };

    out << "digraph DFA {\n";
    out << "  rankdir=LR;\n";
    out << "  node [shape=circle];\n";

    for (const auto &[id, node]: nodes) {

        // Output nodes
        if (node.final > 0) {
            out << "  " << id
                    << " [shape=doublecircle, label=\"" << id
                    << " #" << node.size << "(" << node.final << ")\"];\n";
        } else {
            out << "  " << id
                    << " [label=\"" << id
                    << " #" << node.size << "(" << node.final << ")\"];\n";
        }
        // Output transitions
        for (const auto &[label, edge]: node.edges) {
            out << "  " << id << " -> " << edge.target_nr
                    << " [label=\"" << label << " #" << edge.count << "\"];\n";
        }
    }

    // Initial arrow to the root
    out << "  init [shape=point];\n";
    out << "  init -> " << root_number << ";\n";
    out << "}\n";
}


Model Model::from_apta_json(int id, std::istream &input_stream) {

    json read_apta = json::parse(input_stream);
    Model model(id);

    // Set the root id to -1
    model.root_number = -1;

    // Initialize the locator
    for (auto &i: read_apta["types"]) {
        inputdata_locator::get()->type_from_string(i);
    }
    for (auto &i: read_apta["alphabet"]) {
        inputdata_locator::get()->symbol_from_string(i);
    }

    // Parse the nodes data
    for (int i = 0; i < read_apta["nodes"].size(); ++i) {

        // Create new node from the node data
        json node_json = read_apta["nodes"][i];
        int node_number = node_json["id"];
        int node_size = node_json["size"];
        int node_final = node_json["data"]["total_final"];

        // Create node object
        ModelNode node(node_number, node_size, node_final);

        // Extract the transition counts from the node data
        for (auto& transition_count : node_json["data"]["trans_counts"].items()){
            const std::string symbol_str = transition_count.key();
            const std::string count_str = transition_count.value();
            int symbol = inputdata_locator::get()->symbol_from_string(symbol_str);
            // Create edge object and add it to the source node
            node.add_edge(ModelEdge(symbol, std::stoi(count_str), -1));
        }

        // Pass the ownership of the node to the model
        model.add_node(std::move(node));
    }

    // Parse the edges data to add the targets
    for (int i = 0; i < read_apta["edges"].size(); ++i) {

        // Get the transition symbol
        json edge_json = read_apta["edges"][i];
        std::string symbol_str = edge_json["name"];
        int symbol = inputdata_locator::get()->symbol_from_string(symbol_str);

        // Get the source and the target
        std::string source_string = edge_json["source"];
        std::string target_string = edge_json["target"];

        int source_nr = std::stoi(source_string);
        int target_nr = std::stoi(target_string);

        if (!model.nodes.contains(source_nr)) continue;
        if (!model.nodes.contains(target_nr)) continue;

        // Set the target on the edge
        ModelNode& source = model.nodes.at(source_nr);
        source.edges.at(symbol).target_nr = target_nr;
    }

    return model;
}

