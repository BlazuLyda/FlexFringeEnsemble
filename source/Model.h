//
// Created by Blazej on 07.05.25.
//

#ifndef FLEXFRINGE_MODEL_H
#define FLEXFRINGE_MODEL_H

#include "apta.h"


class ModelNode;

class ModelEdge;

typedef std::map<int, std::unique_ptr<ModelNode>> NodeMap;


/**
 * This is a minimal copy of merged apta that provides the functionality of evaluating sample
 * traces. Used with the ensemble methods to keep track of trained models.
 */
class Model {

private:
	NodeMap nodes; // Owns all nodes of the Model
	ModelNode *root{}; // Non-owning pointer to the root node

public:

	/** constructors and initializers **/
	Model() = default;

	~Model() = default;

	static std::unique_ptr<Model> from_state_merger(state_merger *merger);

	/** Evaluate traces **/
	int evaluate(trace *trace);
};


class ModelNode {

private:
	/** Unique numeric id of the node within the Model */
	int number;
	/** Is this a sink state? Denotes sink type. */
	int sink;
	/** Count of traces in the training set that go through this node. */
	int size;
	/** Count of traces in the training set that finish in this node. */
	int final;
	/** Node transitions. The key is the label. **/
	std::map<int, ModelEdge> edges;


public:
	/** constructors and initializers **/
	ModelNode(int number, int sink, int size, int final) :
			number(number), sink(sink), size(size), final(final) {}

	~ModelNode() = default;

	static std::unique_ptr<ModelNode> from_apta_node(apta_node *apta_node);

	void add_edges(apta_node *apta_node, NodeMap *node_map);

	friend class Model;
};


class ModelEdge {

private:
	/** The numeric label of the transition. Represents a member of the alphabet. **/
	int label;
	/** Count of traces in the training set that follow this edge **/
	int count;
	/** Node the edge finishes at **/
	ModelNode *target;

public:
	/** constructors and initializers **/
	ModelEdge(int label, int count, ModelNode *target) :
			label(label), count(count), target(target) {}

	~ModelEdge() = default;

	inline ModelNode *get_target() { return target; };
};


#endif //FLEXFRINGE_MODEL_H
