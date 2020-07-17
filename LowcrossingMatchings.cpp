#include <iostream>
#include <fstream>
#include <vector>
#include <assert.h>
#include <algorithm> 
#include <time.h>
#include <cmath>
#include <chrono>
#include <random>
#include <math.h>
#include <Eigen/Dense>
#include <thread>

using namespace std;
using namespace Eigen;

#define _uindex unsigned long long //64 bits
#define _sindex long long int      //64 bits
#define _float  double

struct myEdge;
struct myVec;
struct myPlane;

int numThreads = 24;

_uindex _N_DEFAULT = 2000;
_uindex _D_DEFAULT = 5;
_float _SAMPLING_MULTIPLIER = 10.0f;

_uindex _LOG2_N;
_uindex _LOGe_N;
_uindex _pow1dim_N;

_uindex _EDGE_SAMPLED_SEARCH_RADIUS;
_uindex _PLANE_SAMPLED_SEARCH_RADIUS;
_uindex _EDGES_SAMPLED_PER_VERTEX_INITIAL;
_uindex _PLANES_SAMPLED_TOTAL_INITIAL;
_uindex _EDGES_SAMPLED_PER_ITERATION;
_uindex _PLANES_SAMPLED_PER_ITERATION;
_float	_PROPORTION_ITERATIONS;

std::random_device rd;
std::default_random_engine generator(rd());
std::uniform_int_distribution<_uindex> distribution;

void generateCyclicPolytope(vector<myVec> & input, _uindex);
void generateBoxRandom(vector<myVec> & input, _uindex);
void generateUniformRandomGrid(vector<myVec> &, _uindex, _uindex);

struct myVec
{
	vector<_float> coordinates;
	vector<myEdge *> vE; // pointers to all incident active edges 

	myVec() {}
	myVec(vector<_float> & q) { coordinates.reserve(q.size());  for (size_t i = 0; i < q.size(); i++) coordinates.push_back(q[i]); }
	void clearEdges() { vE.clear(); }
};

void setParameters(vector<myVec *> & input)
{
	assert(input.size());
	_uindex _N = input.size();
	_uindex _D = input[0]->coordinates.size();

	_LOG2_N = ((_uindex) log2(_N)) + 1;
	_LOGe_N = ((_uindex) log(_N)) + 1;
	_pow1dim_N = ((_uindex) pow(_N, 1.0f / ((_float)_D))) + 1;

	_EDGE_SAMPLED_SEARCH_RADIUS = ((_uindex) 2.0f * _LOG2_N) + 1;
	_PLANE_SAMPLED_SEARCH_RADIUS = ((_uindex) 2.0f * _LOG2_N) + 1;
	_EDGES_SAMPLED_PER_VERTEX_INITIAL = ((_uindex)_LOG2_N) + 1;
	_PLANES_SAMPLED_TOTAL_INITIAL = ((_uindex)_N * _LOG2_N) + 1;
	_EDGES_SAMPLED_PER_ITERATION = ((_uindex)_SAMPLING_MULTIPLIER * _pow1dim_N * _LOG2_N) + 1;
	_PLANES_SAMPLED_PER_ITERATION = ((_uindex)_SAMPLING_MULTIPLIER * _pow1dim_N * _LOG2_N) + 1;
	_PROPORTION_ITERATIONS = 0.25f;
}

struct myTimer {
	std::chrono::time_point<std::chrono::high_resolution_clock> _currtime;
	myTimer() {}
	void start() { _currtime = std::chrono::high_resolution_clock::now(); }
	double end() {
		std::chrono::duration<double, std::micro> timer_algo_delta = std::chrono::high_resolution_clock::now() - _currtime;
		return timer_algo_delta.count() / 1000000.0f;
	}
};

_float dot(vector<_float> & p, vector<_float> & q)
{
	assert(p.size() == q.size());
	_float sum = 0.0f;
	for (size_t i = 0; i < p.size(); i++) sum += p[i] * q[i];
	return sum;
}

struct myEdge
{
	myVec *p1, *p2;
	_uindex intersection_count; // its weight is (1/2)^intersection_count
	_uindex index_sampling;		// index in the set of all active edges
	_sindex index;				// index in the set of edges of the same weight

	myEdge(myVec *f1 = nullptr, myVec* f2 = nullptr) : p1(f1), p2(f2) {}
};

struct myPlane
{
	vector<_float> N; 			// vector orthogonal to the plane
	_float dist;					// dot product of N and a point on the plane
	_uindex intersection_count;		// plane has weight 2^intersection_count
	_sindex index;					// index in the set of planes with the same weight

	myPlane() { intersection_count = 0; }
	myPlane(vector<myVec *> & points)
	{
		_uindex _D = points[0]->coordinates.size();
		intersection_count = 0;
		MatrixXd A(_D, _D);
		for (size_t i = 0; i < _D; i++)
			for (size_t j = 0; j < _D; j++)
				A(i, j) = points[i]->coordinates[j];
		VectorXd b(_D);
		for (size_t i = 0; i < _D; i++) b(i) = 1.0f;
		//VectorXd ans = A.colPivHouseholderQr().solve(b);
		VectorXd ans = A.partialPivLu().solve(b);
		N.reserve(_D);
		for (size_t i = 0; i < _D; i++) N.push_back(ans(i));
		dist = 1.0f;
	}
	bool side(myVec * & q) { return ((dot(N, q->coordinates) > dist) ? 1 : 0); }
	bool intersectEdge(myEdge * & e) { return (side(e->p1) ^ side(e->p2)); }
	bool intersectEdge(myVec * & p1, myVec * & p2) { return (side(p1) ^ side(p2)); }
};

_uindex max_crossing(vector<myEdge *> & edges, vector<myPlane *> & planes) {
	vector<_uindex> S(planes.size(), 0);
	for (size_t i = 0; i < planes.size(); i += 5)
		for (size_t j = 0; j < edges.size(); j++)
			if (planes[i]->intersectEdge(edges[j])) S[i]++;
	return *max_element(S.begin(), S.end());
}

struct myEdges {
	vector<vector<myEdge *>> CE;
	vector<myEdge *> CE_sampling;
	_uindex minCE_index;

	myEdges() {}
	myEdges(vector<myEdge *> & input_edges) { CE_sampling = input_edges; init_weights(); }
	myEdges(vector<myVec *> & input, size_t num) { sample_edges(input, num, CE_sampling); init_weights(); }

	void init_weights() {
		CE = vector<vector<myEdge *>>(CE_sampling.size(), vector<myEdge *>());
		CE[0] = CE_sampling;
		for (size_t i = 0; i < CE[0].size(); i++) {
			CE[0][i]->p1->vE.push_back(CE[0][i]);
			CE[0][i]->p2->vE.push_back(CE[0][i]);
			CE[0][i]->index = i;
			CE[0][i]->index_sampling = i;
			CE[0][i]->intersection_count = 0;
		}
		minCE_index = 0;
	}

	void sample_edges(vector<myVec *> & input, size_t num, vector<myEdge *> & output)
	{
		output.reserve(input.size() * num);
		for (size_t i = 0; i < input.size(); i++)
			for (size_t j = 0; j < num; j++)
				output.push_back(new myEdge(input[i], input[distribution(generator) % input.size()]));
	}

	myEdge * weighted_sample_edge()
	{
		_uindex curr_sum;
		_uindex W;
		_uindex	p;

		_uindex max_edge_index = min(minCE_index + _EDGE_SAMPLED_SEARCH_RADIUS, (_uindex)( CE.size() - 1) );

		W = 0;
		p = 1;
		for (size_t i = max_edge_index; i >= minCE_index; i--) {
			W += CE[i].size() * p;
			p <<= 1;
			if (i == 0) break;
		}
		p >>= 1;

		_uindex r = distribution(generator) % W;
		curr_sum = 0;
		for (size_t i = minCE_index; i <= max_edge_index; i++) {
			curr_sum += CE[i].size() * p;
			if (curr_sum > r)
				return CE[i][distribution(generator) % CE[i].size()];
			p >>= 1;
		}
		return nullptr;
	}

	myEdge * sample_edge() {
		return CE_sampling[distribution(generator) % CE_sampling.size()]; // Can potentially sample the same edge several times and decrease the weight multiple times instead of one
	}

	void all_edges(vector<myVec *> & input, vector<myEdge *> & output)
	{
		output.resize(input.size() * (input.size() - 1) / 2);
		for (size_t i = 0; i < input.size(); i++)
			for (size_t j = i + 1; j < input.size(); j++)
				output.push_back(new myEdge(input[i], input[j]));
	}

	void increment_weight(myEdge *e)
	{
		CE[e->intersection_count][e->index] = CE[e->intersection_count].back();
		CE[e->intersection_count][e->index]->index = e->index;
		CE[e->intersection_count].pop_back();
		e->intersection_count++;
		if (CE.size() <= e->intersection_count) CE.push_back(vector<myEdge *>());
		CE[e->intersection_count].push_back(e);
		e->index = CE[e->intersection_count].size() - 1;

		while (CE[minCE_index].empty())  minCE_index++;
	}

	void deleteEdge(myEdge * & e)
	{
		if (e->index < 0) return;

		vector<myEdge *> & E_e = CE[e->intersection_count];
		assert(e->index < E_e.size());

		E_e[e->index] = E_e.back();
		E_e[e->index]->index = e->index;
		E_e.pop_back();
		e->index = -1;

		CE_sampling[e->index_sampling] = CE_sampling.back();
		CE_sampling[e->index_sampling]->index_sampling = e->index_sampling;
		CE_sampling.pop_back();
	}

	void delete_adjacent_edges(myEdge *e) {
		for (size_t k = 0; k < e->p1->vE.size(); k++)
			deleteEdge(e->p1->vE[k]);
		for (size_t k = 0; k < e->p2->vE.size(); k++)
			deleteEdge(e->p2->vE[k]);

		while (CE[minCE_index].empty())  minCE_index++;

		e->p1->clearEdges();
		e->p2->clearEdges();
	}
};

struct myPlanes {
	vector<vector<myPlane *>> CH;
	vector<myPlane *> CH_sampling;
	_uindex maxCH_index;

	myPlanes() {}
	myPlanes(vector<myPlane *> & input_planes) { CH_sampling = input_planes;  init_weights(); }
	myPlanes(vector<myVec *> & input, size_t num) { sample_hyperplanes(input, num, CH_sampling); init_weights(); }

	void init_weights() {
		CH = vector<vector<myPlane *>>(CH_sampling.size(), vector<myPlane *>());
		CH[0] = CH_sampling;
		for (size_t i = 0; i < CH[0].size(); i++) {
			CH[0][i]->index = i;
			CH[0][i]->intersection_count = 0;
		}
		maxCH_index = 0;
	}

	void sample_hyperplanes(vector<myVec *> & input, size_t num, vector<myPlane *> & output)
	{
		output.resize(num);
		_uindex _D = input[0]->coordinates.size();
		vector<myVec *> plane(_D);
		for (size_t j = 0; j < num; j++) {
			for (size_t k = 0; k < _D; k++)  plane[k] = input[distribution(generator) % input.size()];
			output[j] = new myPlane(plane);
		}
	}

	_uindex approx_weight(myEdge *e)
	{
		_uindex W = 0;
		_uindex	p = 1;

		_uindex min_plane_index = (maxCH_index > _PLANE_SAMPLED_SEARCH_RADIUS) ? maxCH_index - _PLANE_SAMPLED_SEARCH_RADIUS : 0;

		for (size_t i = min_plane_index; i <= maxCH_index; i++) {
			for (size_t j = 0; j < CH[i].size(); j++)
				if (CH[i][j]->intersectEdge(e)) W += p;
			p <<= 1;
		}
		return W;
	}

	myPlane * weighted_sample_plane()
	{
		_uindex curr_sum;
		_uindex W = 0;
		_uindex	p = 1;

		_uindex min_plane_index = (maxCH_index > _PLANE_SAMPLED_SEARCH_RADIUS) ? maxCH_index - _PLANE_SAMPLED_SEARCH_RADIUS : 0;

		for (size_t i = min_plane_index; i <= maxCH_index; i++) {
			W += CH[i].size() * p;
			p <<= 1;
		}
		p >>= 1;

		_uindex r = distribution(generator) % W;
		curr_sum = 0;
		for (size_t i = maxCH_index; i >= min_plane_index; i--) {
			curr_sum += CH[i].size() * p;
			if (curr_sum > r)
				return CH[i][distribution(generator) % CH[i].size()];
			p >>= 1;
			if (i == 0) break;
		}
		return nullptr;
	}

	myPlane * sample_plane() {
		return CH_sampling[distribution(generator) % CH_sampling.size()]; // Can potentially sample the same plane several times and increase the weight multiple times instead of one
	}

	void increment_weight(myPlane *h) {
		CH[h->intersection_count][h->index] = CH[h->intersection_count].back();
		CH[h->intersection_count][h->index]->index = h->index;
		CH[h->intersection_count].pop_back();
		h->intersection_count++;
		if (CH.size() <= h->intersection_count) CH.push_back(vector<myPlane *>());
		CH[h->intersection_count].push_back(h);
		h->index = CH[h->intersection_count].size() - 1;

		if (!CH[maxCH_index + 1].empty())  maxCH_index++;
	}

	_uindex max_approx_crossing(vector<myEdge *> & edges) {
		return max_crossing(edges, CH[maxCH_index]);
	}

	void add_max_planes(vector<myPlane *> & output) {
		for (size_t i = 0; i < CH[maxCH_index].size(); i++)
			output.push_back(CH[maxCH_index][i]);
	}
};


void computeSpanningTree_iteration(vector<myVec *> & input, myPlanes * & testplanes, vector<myEdge *> & output, vector<myPlane *> & heavy_planes)
{
	setParameters(input);
	testplanes->init_weights();

	for (size_t i = 0; i < input.size(); i++) input[i]->clearEdges();
	myEdges *presamplededges = new myEdges(input, _EDGES_SAMPLED_PER_VERTEX_INITIAL);

	_uindex num_iterations = (_uindex)(input.size() * _PROPORTION_ITERATIONS);
	for (size_t i = 0; i < num_iterations; i++) {
		myEdge *random_edge = presamplededges->weighted_sample_edge();
		myPlane *random_hyperplane = testplanes->weighted_sample_plane();
		assert(random_edge); assert(random_hyperplane);

		for (size_t r = 0; r < _EDGES_SAMPLED_PER_ITERATION; r++) {
			myEdge *e = presamplededges->sample_edge();
			if (random_hyperplane->intersectEdge(e)) presamplededges->increment_weight(e);
		}
		presamplededges->delete_adjacent_edges(random_edge);

		for (size_t r = 0; r < _PLANES_SAMPLED_PER_ITERATION; r++) {
			myPlane *h = testplanes->sample_plane();
			if (h->intersectEdge(random_edge)) testplanes->increment_weight(h);
		}
		output.push_back(random_edge);
	}

	testplanes->add_max_planes(heavy_planes);
}

_uindex computeSpanningTree(vector<myVec *> input, vector<myEdge *> & output)
{
	myTimer _t;

	setParameters(input);

	cout << "Started testplane generation.\n"; _t.start();
	myPlanes *testplanes = new myPlanes(input, _PLANES_SAMPLED_TOTAL_INITIAL);
	cout << "\t Finished testplane generation in time: " << _t.end() << endl;

	vector<myPlane *> heavy_planes;
	output.clear();

	size_t iters = 0;
	while (input.size() >= 100) {
		cout << "  Iteration " << ++iters << endl; _t.start();
		computeSpanningTree_iteration(input, testplanes, output, heavy_planes);
		cout << "  Finished in time: " << _t.end() << endl;

		size_t i = 0;
		while (i < input.size()) {
			if (!input[i]->vE.size()) {
				input[i] = input.back();
				input.pop_back();
			}
			else i++;
		}
	}

	return max_crossing(output, heavy_planes);
}

 
int main(int argc, char **argv)
{
	_uindex _N = _N_DEFAULT;
	_uindex _D = _D_DEFAULT;
	string input_distrib =  "Moment";

	for (int i = 0; i < argc; i++)
	{
		string s = argv[i];
		if (s == "-N") _N = atoi(argv[++i]);
		else if (s == "-D") _D = atoi(argv[++i]);
		else if (s == "-M") _SAMPLING_MULTIPLIER = atof(argv[++i]);
		else if (s == "-I") input_distrib = string(argv[++i]);	
	}


	distribution = std::uniform_int_distribution<_uindex>(0, _N * _N * _N);
	srand( (unsigned int) time(NULL));

	vector<myVec> vector_of_points(_N);

	if (input_distrib == "Moment") generateCyclicPolytope(vector_of_points, _D); 
	else if (input_distrib == "Grid") generateUniformRandomGrid(vector_of_points, _N, _D);
	else generateBoxRandom(vector_of_points, _D);


	vector<myVec *>  input(_N);
	for (size_t i = 0; i < _N; i++) input[i] = &vector_of_points[i];

	cout << "n: " << _N << ", dimension: " << _D << endl;

	vector<myEdge *>  output;
	_uindex crossing_number;

	crossing_number = computeSpanningTree(input, output);
	cout << "Final matching size: " << output.size() << endl;
	cout << "Our algo: \n";
	cout << "\tCrossing number: " << crossing_number << ",  approximation ratio : " << (float)crossing_number / pow(_N, 1 - 1.0f / (float)_D) << endl;
}

void generateUniformGrid(vector<myVec> & vector_of_points, _uindex _D, _uindex eachdim)
{
	if (_D == 0) {
		vector_of_points.push_back(myVec());
		return;
	}

	vector<myVec> vec_onelowerdim; 
	generateUniformGrid(vec_onelowerdim, _D - 1, eachdim);
	_float onebyeachdim = 1.0f / (_float)eachdim; 
	for (size_t i = 0; i < vec_onelowerdim.size(); i++) {
		for (size_t j = 0; j < eachdim; j++)
		{
			myVec p = vec_onelowerdim[i];
			p.coordinates.push_back(j * onebyeachdim); 
			vector_of_points.push_back(p);
		}
	}
}

void generateBoxRandom(vector<myVec> & vector_of_points, _uindex _D)
{
	for (size_t i = 0; i < vector_of_points.size(); i++)
		for (size_t j = 0; j < _D; j++) vector_of_points[i].coordinates.push_back(rand() / ((_float)RAND_MAX));
}

void generateUniformRandomGrid(vector<myVec> & vector_of_points, _uindex _N, _uindex _D)
{
	vector_of_points.clear(); 

	_uindex eachdim = ((_uindex)pow(_N, 1.0f / ((_float)_D))) + 1;
	_float onebyeachdim = 1.0f / (_float)eachdim;

	generateUniformGrid(vector_of_points, _D, eachdim); 

	for (size_t i = 0; i < vector_of_points.size(); i++)
		for (size_t j = 0; j < vector_of_points[i].coordinates.size(); j++)
			vector_of_points[i].coordinates[j] += onebyeachdim * (rand() / ((_float)RAND_MAX));
}


void generateCyclicPolytope(vector<myVec> & vector_of_points, _uindex _D)
{
	_float _t = 0.0f;
	for (size_t i = 0; i < vector_of_points.size(); i++) {
		for (size_t j = 0; j < _D; j++) vector_of_points[i].coordinates.push_back(pow(_t + 0.1f * rand() / ((_float)RAND_MAX), j + 1));
		_t += 1.0f;
	}
}

