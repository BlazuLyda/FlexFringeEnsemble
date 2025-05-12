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

	int id = 0;
	NodeMap nodes; // Owns all nodes of the Model
	ModelNode* root{}; // Non-owning pointer to the root node

public:

	/** constructors and initializers **/
	explicit Model(const int id): id(id) {};

	~Model() = default;

	static std::unique_ptr<Model> from_state_merger(int id, state_merger* merger);
	static std::unique_ptr<Model> from_apta_json(int id, std::istream& input_stream);


	/**
	 * Returns the probability of the trace occurring in the model.
	 * @param trace the trace to be evaluated
	 * @return 0 if impossible, >0 if trace ends up in an accepting state
	 */
	double evaluate(trace* trace) const;

	int get_id() const {
		return id;
	}

	void write_dot(std::ostream& output) const;
};

class ModelEdge {

	/** The numeric label of the transition. Represents a member of the alphabet. **/
	int symbol;
	/** Count of traces in the training set that follow this edge **/
	int count;
	/** Node the edge finishes at **/
	ModelNode* target;

public:
	/** constructors and initializers **/
	ModelEdge(const int symbol, const int count, ModelNode* target) :
			symbol(symbol), count(count), target(target) {}

	~ModelEdge() = default;

	ModelNode* get_target() const {
		return target;
	}

	friend class Model;
};

class ModelNode {

	/** Unique numeric id of the node within the Model */
	int number;
	/** Is this a sink state? Denotes sink type. */
	int sink = 0;
	/** Count of traces in the training set that go through this node. */
	int size;
	/** Count of traces in the training set that finish in this node. */
	int final;
	/** Node transitions. The key is the label. **/
	std::map<int, ModelEdge> edges;


public:
	/** constructors and initializers **/
	ModelNode(const int number, const int size, const int final) :
			number(number), size(size), final(final) {}

	~ModelNode() = default;

	static std::unique_ptr<ModelNode> from_apta_node(apta_node* apta_node);

	void add_edges(apta_node* apta_node, NodeMap* node_map);

	[[nodiscard]] std::optional<std::reference_wrapper<const ModelEdge>> follow(const int symbol) const {
		const auto it = edges.find(symbol);
		if (it == edges.end()) {
			return std::nullopt;
		}
		return std::cref(it->second);
	}

	friend class Model;
};

#endif //FLEXFRINGE_MODEL_H
