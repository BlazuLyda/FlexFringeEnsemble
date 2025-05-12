//
// Created by blaze on 07.05.25.
//

#include "Model.h"
#include "alergia.h"
#include "input/inputdatalocator.h"
#include "utility/loguru.hpp"


double Model::evaluate(trace* trace) const {

    auto node_it = root;
    auto trace_it = trace->get_head();
    double prob = 1;

    // std::cout << "Starting evaluation on model " << id << std::endl;
    // Iterate through the trace while keeping the position in the graph
    while (trace_it != nullptr && !trace_it->is_final()) {
        // Get the transition corresponding to the trace symbol
        auto transition = node_it->follow(trace_it->get_symbol());
        // If trace follows a non-existing transition
        if (!transition) {
            // std::cout << "Followed a non-existing path: " << trace_it->get_symbol() << std::endl;
            return 0;
        }
        // Update the probability
        prob *= static_cast<double>(transition.value().get().count) / node_it->size;
        // Move the iterators by one step
        node_it = transition.value().get().get_target();
        // std::cout << "Followed to node: " << node_it->number << ", with path: " << trace_it->get_symbol() << std::endl;
        trace_it = trace_it->future();
    }

    // Now check if the trace ends in an accepting state
    // std::cout << "Finished in node: " << node_it->number << " with final count: " << node_it->final << std::endl;
    if (node_it->final == 0) return 0;
    prob *= static_cast<double>(node_it->final) / node_it->size;
    return prob;
}

std::unique_ptr<Model> Model::from_state_merger(int id, state_merger* merger) {
    // Get the apta from the merger
    apta* apta = merger->get_aut();
    apta_node* root = apta->get_root();

    // Create a new model
    auto model = std::make_unique<Model>(id);

    // Copy all the apta nodes
    for (auto ait = merged_APTA_iterator(root); *ait != nullptr; ++ait) {
        // Create a new node
        apta_node* apta_node = *ait;
        auto new_node = ModelNode::from_apta_node(apta_node);
        // Add the node to the model
        model->nodes.insert({new_node->number, std::move(new_node)});
    }
    // Set the root pointer
    model->root = model->nodes.at(root->get_number()).get();

    // Copy all the apta transitions
    for (auto ait = merged_APTA_iterator(root); *ait != nullptr; ++ait) {
        // Get the parent node
        apta_node* node = *ait;
        ModelNode* model_node = model->nodes.at(node->get_number()).get();
        // Add the edges
        model_node->add_edges(node, &model->nodes);
    }
    return model;
}

std::unique_ptr<ModelNode> ModelNode::from_apta_node(apta_node* node) {
    auto new_node = std::make_unique<ModelNode>(
        node->get_number(),
        node->get_size(), // TODO: check if these are correct, or should I always call find()
        node->get_final()
    );
    return new_node;
}

void ModelNode::add_edges(apta_node* node, NodeMap* node_map) {
    auto* node_data = dynamic_cast<alergia_data *>(node->get_data());

    // Iterate through the edges and add them
    for (auto it = node->guards_start(); it != node->guards_end(); ++it) {
        // Get edge data
        int label = it->first;
        const int edge_count = node_data->count(it->first);
        apta_node* child = it->second->get_target()->find();
        ModelNode* child_node = node_map->find(child->get_number())->second.get();

        // Construct and add the edge
        auto edge = ModelEdge(label, edge_count, child_node);
        this->edges.insert({label, edge});
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

    for (const auto &[id, node_ptr]: nodes) {
        ModelNode* node = node_ptr.get();

        // Output nodes
        if (node->final > 0) {
            out << "  " << id
                    << " [shape=doublecircle, label=\"" << id
                    << " #" << node->size << "(" << node->final << ")\"];\n";
        } else {
            out << "  " << id
                    << " [label=\"" << id
                    << " #" << node->size << "(" << node->final << ")\"];\n";
        }
        // Output transitions
        for (const auto &[label, edge]: node->edges) {
            ModelNode* target = edge.target;
            if (target) {
                out << "  " << id << " -> " << target->number
                        << " [label=\"" << label << " #" << edge.count << "\"];\n";
            }
        }
    }

    // Initial arrow to the root
    if (root) {
        out << "  init [shape=point];\n";
        out << "  init -> " << root->number << ";\n";
    }
    out << "}\n";
}


std::unique_ptr<Model> Model::from_apta_json(int id, std::istream &input_stream) {

    json read_apta = json::parse(input_stream);
    auto model = std::make_unique<Model>(id);

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

        auto node = std::make_unique<ModelNode>(node_number, node_size, node_final);
        // If id is -1 set the root
        if (node->number == -1) {
            model->root = node.get();
        }

        // Extract the transition counts from the node data
        for (auto& transition_count : node_json["data"]["trans_counts"].items()){
            const std::string symbol_str = transition_count.key();
            const std::string count_str = transition_count.value();
            int symbol = inputdata_locator::get()->symbol_from_string(symbol_str);
            const auto edge = ModelEdge(symbol, std::stoi(count_str), nullptr);
            node->edges.insert({symbol, edge});
        }

        // Transfer the ownership to the model object
        model->nodes.insert({node_json["id"], std::move(node)});
    }

    // If root node not found throw exception
    if (model->root == nullptr) {
        std::cerr << "The model root is not specified\n";
        throw std::runtime_error("The model root is not specified");
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

        if (!model->nodes.contains(source_nr)) continue;
        if (!model->nodes.contains(target_nr)) continue;

        ModelNode* source = model->nodes.at(source_nr).get();
        ModelNode* target = model->nodes.at(target_nr).get();

        // Set the target on the edge
        source->edges.at(symbol).target = target;
    }

    return model;
}

