//
// Created by Blazej on 07.05.25.
//

#ifndef FLEXFRINGE_MODEL_H
#define FLEXFRINGE_MODEL_H

#include "apta.h"


class ModelNode;

class ModelEdge {

	/** The numeric label of the transition. Represents a member of the alphabet. **/
	int symbol;
	/** Count of traces in the training set that follow this edge **/
	int count;
	/** Number of the node that the edge finishes at **/
	int target_nr;

public:
	/** constructors and initializers **/
	ModelEdge(const int symbol, const int count, const int target_nr) :
			symbol(symbol), count(count), target_nr(target_nr) {}

	~ModelEdge() = default;

	[[nodiscard]] int get_target() const {
		return target_nr;
	}

	friend class Model;
	friend class ModelNode;
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
	std::map<int, ModelEdge> edges = {};


public:
	/** constructors and initializers **/
	ModelNode(const int number, const int size, const int final) :
			number(number), size(size), final(final) {}

	~ModelNode() = default;

	static ModelNode from_apta_node(apta_node& node);

	void add_edges_from_apta(apta_node& node);

	void add_edge(ModelEdge&& edge) {
		edges.insert({edge.symbol, edge});
	}

	[[nodiscard]] std::optional<std::reference_wrapper<const ModelEdge>> follow(const int symbol) const {
		const auto it = edges.find(symbol);
		if (it == edges.end()) {
			return std::nullopt;
		}
		return std::cref(it->second);
	}

	friend class Model;
};

struct ModelTrace {
	std::vector<int> symbols;
	double prob = 1;
	unsigned int hash = 0;

	unsigned int get_hash() {
		if (hash != 0) return hash;
		for (const int sym: symbols) {
			constexpr unsigned int base = 131;
			hash = hash * base + sym;
		}
		return hash;
	}
};

/**
 * This is a minimal copy of merged apta that provides the functionality of evaluating sample
 * traces. Used with the ensemble methods to keep track of trained models.
 */
class Model {

	int id = 0;
	std::map<int, ModelNode> nodes = {}; // Owns all nodes of the Model
	int root_number = -1; // Number of the root

public:

	/** constructors and initializers **/
	Model() = default;

	~Model() = default;

	static Model from_state_merger(state_merger* merger);
	static Model from_apta_json(std::istream& input_stream);


	/**
	 * Returns the probability of the trace occurring in the model.
	 * @param trace the trace to be evaluated
	 * @return 0 if impossible, >0 if trace ends up in an accepting state
	 */
	[[nodiscard]] double predict(trace* trace) const;

	/**
	 * Returns the probability of the trace occurring in the model.
	 * @param trace the model trace to be evaluated
	 * @return 0 if impossible, >0 if trace ends up in an accepting state
	 */
	[[nodiscard]] double predict(const ModelTrace& trace) const;

	[[nodiscard]] int get_id() const {
		return id;
	}

	void set_id(const int id_to_set) {
		id = id_to_set;
	}

	void add_node(ModelNode&& node) {
		nodes.insert({node.number, std::move(node)});
	}

	[[nodiscard]] const ModelNode& get_node(const int number) const {
		return nodes.at(number);
	}

	void write_dot(std::ostream& output) const;

	/**
	 * Generates a random trace from the model using a random walk.
	 * @return generated trace
	 */
	[[nodiscard]] ModelTrace generate_trace() const;

	/**
	 * Computes the average cross-entropy on the provided sample set for this model.
	 * @param traces sample set of traces
	 * @return cross-entropy divided by the sample set size
	 */
	[[nodiscard]] double compute_diff(const std::vector<ModelTrace> &traces) const;
};

#endif //FLEXFRINGE_MODEL_H
