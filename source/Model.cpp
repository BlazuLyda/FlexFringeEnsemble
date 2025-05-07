//
// Created by blaze on 07.05.25.
//

#include "Model.h"
#include "alergia.h"

std::unique_ptr<Model> Model::from_state_merger(state_merger *merger) {

	// Get the apta from the merger
	apta *apta = merger->get_aut();
	apta_node *root = apta->get_root();

	// Create a new model
	auto model = std::make_unique<Model>();

	// Copy all the apta nodes
	for (merged_APTA_iterator ait = merged_APTA_iterator(root); *ait != nullptr; ++ait) {

		// Create a new node
		apta_node *apta_node = *ait;
		auto new_node = ModelNode::from_apta_node(apta_node);

		// Add the node to the model
		model->nodes.insert({new_node->number, std::move(new_node)});
	}

	// Copy all the apta transitions
	for (merged_APTA_iterator ait = merged_APTA_iterator(root); *ait != nullptr; ++ait) {

		// Get the parent node
		apta_node *node = *ait;
		ModelNode *model_node = model->nodes.at(node->get_number()).get();

		// Add the edges
		model_node->add_edges(node, &model->nodes);
	}
	return model;
}

std::unique_ptr<ModelNode> ModelNode::from_apta_node(apta_node *node) {

	auto new_node = std::make_unique<ModelNode>(
			node->find()->get_number(),
			node->sink_type(),
			node->get_number(), // TODO: check if these are correct, or should I always call find()
			node->get_final()
	);
	return new_node;
}

void ModelNode::add_edges(apta_node *node, NodeMap *node_map) {

	auto *node_data = dynamic_cast<alergia_data *>(node->get_data());

	// Iterate through the edges and add them
	for (auto it = node->guards_start(); it != node->guards_end(); ++it) {

		// Get edge data
		int label = it->first;
		int edge_count = node_data->count(it->first);
		apta_node* child = it->second->get_target()->find();
		ModelNode *child_node = node_map->find(child->get_number())->second.get();

		// Construct and add the edge
		ModelEdge edge = ModelEdge(label, edge_count, child_node);
		this->edges.insert({label, edge});
	}
}
