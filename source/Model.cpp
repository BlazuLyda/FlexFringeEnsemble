//
// Created by blaze on 07.05.25.
//

#include "Model.h"
#include "alergia.h"
#include "utility/loguru.hpp"

std::unique_ptr<Model> Model::from_state_merger(state_merger* merger) {

	// Get the apta from the merger
	apta* apta = merger->get_aut();
	apta_node* root = apta->get_root();

	// Create a new model
	auto model = std::make_unique<Model>();

	// Copy all the apta nodes
	for (merged_APTA_iterator ait = merged_APTA_iterator(root); *ait != nullptr; ++ait) {

		// Create a new node
		apta_node* apta_node = *ait;
		auto new_node = ModelNode::from_apta_node(apta_node);
		// Add the node to the model
		model->nodes.insert({new_node->number, std::move(new_node)});
	}

	// Copy all the apta transitions
	for (merged_APTA_iterator ait = merged_APTA_iterator(root); *ait != nullptr; ++ait) {

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
			node->find()->get_number(),
			node->sink_type(),
			node->get_number(), // TODO: check if these are correct, or should I always call find()
			node->get_final()
	);
	return new_node;
}

void ModelNode::add_edges(apta_node* node, NodeMap* node_map) {

	auto* node_data = dynamic_cast<alergia_data*>(node->get_data());

	// Iterate through the edges and add them
	for (auto it = node->guards_start(); it != node->guards_end(); ++it) {

		// Get edge data
		int label = it->first;
		int edge_count = node_data->count(it->first);
		apta_node* child = it->second->get_target()->find();
		ModelNode* child_node = node_map->find(child->get_number())->second.get();

		// Construct and add the edge
		ModelEdge edge = ModelEdge(label, edge_count, child_node);
		this->edges.insert({label, edge});
	}
}


void Model::write_dot(std::ostream& out) const {
	if (!out) {
		LOG_S(ERROR) << "The output is not specified\n";
		return;
	};

	out << "digraph DFA {\n";
	out << "  rankdir=LR;\n";
	out << "  node [shape=circle];\n";

	for (const auto& [id, node_ptr] : nodes) {
		ModelNode* node = node_ptr.get();

		// Mark final/sink states differently
		if (node->sink) {
			out << "  " << id
			    << " [shape=doublecircle, label=\"" << id
			    << " #" << node->size << "(" << node->final << ")"
			    << " (sink " << node->sink << ")\"];\n";
		} else {
			out << "  " << id
			    << " [label=\"" << id
			    << " #" << node->size << "(" << node->final << ")\"];\n";
		}

		for (const auto& [label, edge] : node->edges) {
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
