#include <iostream>
#include <fstream>
#include <iomanip>
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

_uindex _NUMTHREADS = 24;
_uindex _N_DEFAULT = 5000;
_uindex _D_DEFAULT = 4;
_uindex _THRESHOLD = 0;

_uindex GLOBAL_N;
_uindex GLOBAL_D;
string GLOBAL_DISTRIB;
string GLOBAL_FILENAME;
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
void generateCircle(vector<myVec> & input, _uindex);
void generateSineCurve(vector<myVec> & input, _uindex);
void saveFile(vector<myVec> &, string & fname);
void saveFile(vector<myEdge *> & output, string & fname);



/*

OUR STRUCTURES

*/

struct myVec				// structure for points
{
	vector<_float> coordinates;
	vector<myEdge *> vE; // pointers to all incident active edges 

	myVec() {}
	myVec(vector<_float> & q) { coordinates.reserve(q.size());  for (size_t i = 0; i < q.size(); i++) coordinates.push_back(q[i]); }
	void clearEdges() { vE.clear(); }
};



struct myEdge 		// structure for edges
{
	myVec *p1, *p2;
	_uindex intersection_count; // its weight is (1/2)^intersection_count
	_uindex index_sampling;		// index in the set of all active edges
	_sindex index;				// index in the set of edges of the same weight
	_sindex thread;				// index of the thread working with this edge.

	myEdge(myVec *f1 = nullptr, myVec* f2 = nullptr) : p1(f1), p2(f2) {}
};

_float dot(vector<_float> & p, vector<_float> & q)	//dot product of two vectors
{
	assert(p.size() == q.size());
	_float sum = 0.0f;
	for (size_t i = 0; i < p.size(); i++) sum += p[i] * q[i];
	return sum;
}

struct myPlane 			// structure for hyperplanes
{
	vector<_float> N; 				// vector orthogonal to the plane
	_float dist;					// dot product of N and a point on the plane
	_uindex intersection_count;		// plane has weight 2^intersection_count
	_sindex index;					// index in the set of planes with the same weight

	myPlane() { intersection_count = 0; }

	myPlane(vector<myVec *> & points)  // constructing a hyperplane through points
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

_uindex max_crossing(vector<myEdge *> & edges, vector<myPlane *> & planes) // computes max crossing number of 'edges' with respect to 'planes'
{
	vector<_uindex> S(planes.size(), 0);
	for (size_t i = 0; i < planes.size(); i += 5)
		for (size_t j = 0; j < edges.size(); j++)
			if (planes[i]->intersectEdge(edges[j])) S[i]++;
	return *max_element(S.begin(), S.end());
}



void setParameters(vector<myVec *>  input)
{
	assert(input.size());
	_uindex _N = input.size();
	_uindex _D = input[0]->coordinates.size();

	_LOG2_N = ((_uindex)log2(_N)) + 1;
	_LOGe_N = ((_uindex)log(_N)) + 1;
	_pow1dim_N = ((_uindex)pow(_N, 1.0f / ((_float)_D))) + 1;

	_EDGE_SAMPLED_SEARCH_RADIUS = ((_uindex) 2.0f * _LOG2_N) + 1;
	_PLANE_SAMPLED_SEARCH_RADIUS = ((_uindex) 2.0f * _LOG2_N) + 1;

	_EDGES_SAMPLED_PER_VERTEX_INITIAL = ((_uindex) 4 * _LOG2_N) + 1;
	_PLANES_SAMPLED_TOTAL_INITIAL = ((_uindex)_N * _LOG2_N) + 1;


	_EDGES_SAMPLED_PER_ITERATION = ((_uindex)  10 *_pow1dim_N * pow(_LOG2_N, 2) ) + 1;
	_PLANES_SAMPLED_PER_ITERATION = ((_uindex)  1 * _pow1dim_N * pow(_LOG2_N, 2) ) + 1;

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



struct myEdges {		// structure for the edge set
	vector<vector<myEdge *>> CE;    // edges partitioned by weight for weighted sampling
	vector<myEdge *> CE_sampling;   // the list of active edges from which to uniformly sample
	_uindex minCE_index;

	myEdges() {}
	myEdges(vector<myEdge *> & input_edges, _uindex thread_no = 0) { CE_sampling = input_edges; init_weights(thread_no); }
	myEdges(vector<myVec *> & input, size_t num, _uindex thread_no) { sample_initial_edges(input, num, CE_sampling); init_weights(thread_no); }

	void init_weights(_uindex thread_no) {
		CE = vector<vector<myEdge *>>(CE_sampling.size(), vector<myEdge *>());
		CE[0] = CE_sampling;
		for (size_t i = 0; i < CE[0].size(); i++) {
			CE[0][i]->p1->vE.push_back(CE[0][i]);
			CE[0][i]->p2->vE.push_back(CE[0][i]);
			CE[0][i]->index = i;
			CE[0][i]->index_sampling = i;
			CE[0][i]->intersection_count = 0;
			CE[0][i]->thread = thread_no;
		}
		minCE_index = 0;
	}

	void sample_initial_edges(vector<myVec *> & input, size_t num, vector<myEdge *> & output)	 // uniformly samples 'num' many edges per vertex
	{
		output.reserve(input.size() * num);
		for (size_t i = 0; i < input.size(); i++)
			for (size_t j = 0; j < num; j++) {
				size_t k;
				do {
					k = distribution(generator) % input.size();
				} while (k == i);
				output.push_back(new myEdge(input[i], input[k]));
			}
	}

	myEdge * weighted_sample_edge() // samples one edge according to current weights
	{
		_uindex curr_sum;
		_uindex W;	// total weight of edges
		_uindex	p;

		_uindex max_edge_index = min(minCE_index + _EDGE_SAMPLED_SEARCH_RADIUS, (_uindex)(CE.size() - 1));

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

	myEdge * sample_edge() // samples one edge uniformly at random
	{
		return CE_sampling[distribution(generator) % CE_sampling.size()]; 
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

		while (CE[minCE_index].empty())  minCE_index++;
	}



	void test_hyperplane(myPlane *h, _uindex num)	// sample num many edges and for each, increment its weight if it is crossed by h
	{
		for (size_t i = 0; i < num; i++) {
			myEdge *e = sample_edge();
			if (h->intersectEdge(e)) increment_weight(e);
		}
	}
};

struct myPlanes 		// structure for hyperplane set
{
	vector<vector<myPlane *>> CH;	//  hyperplanes partitioned by weight for weighted sampling
	vector<myPlane *> CH_sampling;	// vector of all test hyperplanes
	_uindex maxCH_index;

	myPlanes() {}
	myPlanes(vector<myPlane *> & input_planes) { CH_sampling = input_planes;  init_weights(); }
	myPlanes(vector<myVec *> & input, size_t num) { sample_initial_hyperplanes(input, num, CH_sampling); init_weights(); }

	void init_weights() {
		CH = vector<vector<myPlane *>>(CH_sampling.size(), vector<myPlane *>());
		CH[0] = CH_sampling;
		for (size_t i = 0; i < CH[0].size(); i++) {
			CH[0][i]->index = i;
			CH[0][i]->intersection_count = 0;
		}
		maxCH_index = 0;
	}

	void sample_initial_hyperplanes(vector<myVec *> & input, size_t num, vector<myPlane *> & output)	 // samples 'num' many hyperplanes through input points
	{
		output.resize(num);
		_uindex _D = input[0]->coordinates.size();
		vector<myVec *> plane(_D);
		for (size_t j = 0; j < num; j++) {
			for (size_t k = 0; k < _D; k++)  plane[k] = input[distribution(generator) % input.size()];
			output[j] = new myPlane(plane);
		}
	}

	

	myPlane * weighted_sample_plane()	// samples one hyplerplane according to current weights
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

	myPlane * sample_plane() // samples one hyperplane uniformly
	{
		return CH_sampling[distribution(generator) % CH_sampling.size()]; 
	}

	void increment_weight(myPlane *h) 
	{
		CH[h->intersection_count][h->index] = CH[h->intersection_count].back();
		CH[h->intersection_count][h->index]->index = h->index;
		CH[h->intersection_count].pop_back();
		h->intersection_count++;
		if (CH.size() <= h->intersection_count) CH.push_back(vector<myPlane *>());
		CH[h->intersection_count].push_back(h);
		h->index = CH[h->intersection_count].size() - 1;

		if (!CH[maxCH_index + 1].empty())  maxCH_index++;
	}

	
	void add_max_planes(vector<myPlane *> & output) 
	{
		for (size_t i = 0; i < CH[maxCH_index].size(); i++)
			output.push_back(CH[maxCH_index][i]);
	}

	void test_edge(myEdge *e, _uindex num)	// sample num many hyperplanes and for each, increment its weight if it crosses e
	{
		for (size_t r = 0; r < num; r++) 
		{
			myPlane *h = sample_plane();
			if (h->intersectEdge(e)) increment_weight(h);
		}
	}
};



myEdge * weighted_sample_edge(vector<myEdges *> & presamplededges)
{
	myEdge *e = nullptr;
	while (!e)
		e = presamplededges[distribution(generator) % presamplededges.size()]->weighted_sample_edge();
	return e;
}

myPlane * weighted_sample_plane(vector<myPlanes *> &  testplanes)
{
	myPlane *h = nullptr;
	while (!h)
		h = testplanes[distribution(generator) % testplanes.size()]->weighted_sample_plane();
	return h;
}




/*

MATCHING ALGORITHM

*/

void computeMatching_iteration(vector<myVec *> & input, vector<myPlanes *> & testplanes, vector<myEdge *> & output, vector<myPlane *> & heavy_planes)
{
	setParameters(input);

	_uindex num_edgethreads = min((_uindex)(_EDGES_SAMPLED_PER_ITERATION / 2000.0f + 1), _NUMTHREADS);

	vector<myEdges *> presamplededges(num_edgethreads);


	for (size_t i = 0; i < input.size(); i++) input[i]->clearEdges();
	for (size_t t = 0; t < testplanes.size(); t++) testplanes[t]->init_weights();

	for (size_t t = 0; t < presamplededges.size(); t++) presamplededges[t] = new myEdges(input, (_uindex)(_EDGES_SAMPLED_PER_VERTEX_INITIAL / (_float)presamplededges.size()+1), t);

	_uindex num_iterations = (_uindex)(input.size() * _PROPORTION_ITERATIONS);

	_uindex curr_threshold = (_uindex)(0.5f * num_iterations); // number of initial warm-up iterations
	

	for (size_t i = 0; i < num_iterations + curr_threshold; i++) {

		if (i == curr_threshold)	// initalize plane weights after warm-up
			for (size_t t = 0; t < testplanes.size(); t++) testplanes[t]->init_weights();

		myEdge *random_edge = weighted_sample_edge(presamplededges);
		myPlane *random_hyperplane = weighted_sample_plane(testplanes);
		assert(random_edge); assert(random_hyperplane);

		vector<thread> Threads;

		for (size_t t = 0; t < presamplededges.size(); t++)
			Threads.push_back(thread(&myEdges::test_hyperplane, presamplededges[t], random_hyperplane, (_uindex)(_EDGES_SAMPLED_PER_ITERATION / (_float)presamplededges.size())));

		if (i >= curr_threshold) 
		{
			for (size_t t = 0; t < testplanes.size(); t++)
				Threads.push_back(thread(&myPlanes::test_edge, testplanes[t], random_edge, (_uindex)(_PLANES_SAMPLED_PER_ITERATION / (_float)testplanes.size())));
		}

		for (size_t t = 0; t < Threads.size(); t++) Threads[t].join();


		if (i >= curr_threshold)
		{
			for (size_t k = 0; k < random_edge->p1->vE.size(); k++)
				presamplededges[random_edge->p1->vE[k]->thread]->deleteEdge(random_edge->p1->vE[k]);
			for (size_t k = 0; k < random_edge->p2->vE.size(); k++)
				presamplededges[random_edge->p2->vE[k]->thread]->deleteEdge(random_edge->p2->vE[k]);
			random_edge->p1->clearEdges();
			random_edge->p2->clearEdges();
			presamplededges[random_edge->thread]->deleteEdge(random_edge);
			output.push_back(random_edge);
		}
		

	}

	for (size_t t = 0; t < testplanes.size(); t++)
		testplanes[t]->add_max_planes(heavy_planes);

}

_uindex computeMatching(vector<myVec *> input, vector<myEdge *> & output, _float & timetaken)
{
	myTimer _t;
	//_float timer = 0.0f;

	setParameters(input);

	_uindex num_planethreads = min((_uindex)(_PLANES_SAMPLED_PER_ITERATION / 2000.0f + 1), _NUMTHREADS);

	vector<myPlanes *> testplanes(num_planethreads);

	//cout << "Started testplane generation.\n"; _t.start();

	for (size_t t = 0; t < testplanes.size(); t++)
		testplanes[t] = new myPlanes(input, (_uindex)(_PLANES_SAMPLED_TOTAL_INITIAL / (_float)testplanes.size()));

	//cout << "\tFinished testplane generation in time: " << _t.end() << endl;

	vector<myPlane *> heavy_planes;
	output.clear();

	size_t iters = 0;
	_t.start();
	while (input.size() >= 40) {

		//cout << "  Iteration " << ++iters << endl; _t.start();

		computeMatching_iteration(input, testplanes, output, heavy_planes);

		//cout << "  Finished in time: " << _t.end() << endl;
		//timer += _t.end(); 
		{
			size_t i = 0;
			while (i < input.size()) {
				if (!input[i]->vE.size()) {
					input[i] = input.back();
					input.pop_back();
				}
				else i++;
			}
		}
	}

	timetaken = _t.end();
	return max_crossing(output, heavy_planes);
}


int main(int argc, char **argv)
{
	_uindex _N = _N_DEFAULT;
	_uindex _D = _D_DEFAULT;
	bool output_tofile = false;
	string input_distrib = string("Uniform");

	_uindex crossing_number = 0;
	 _float timetaken = 0;	
			 
			 

	for (int i = 0; i < argc; i++)
	{
		string s = argv[i];
		if (s == "-N") _N = atoi(argv[++i]);
		else if (s == "-D") _D = atoi(argv[++i]);
		else if (s == "-I") input_distrib = string(argv[++i]);
		else if (s == "-T") _THRESHOLD = atoi(argv[++i]);
		else if (s == "-F") output_tofile = true;
		else if (s == "-THREADS") _NUMTHREADS = atoi(argv[++i]);
		else if (s == "-FN") GLOBAL_FILENAME = string(argv[++i]);
	}

	GLOBAL_N = _N;
	GLOBAL_D = _D;
	GLOBAL_DISTRIB = input_distrib;

		

		distribution = std::uniform_int_distribution<_uindex>(0, _N * _N * _N);
		srand((unsigned int)time(NULL));

		vector<myVec> vector_of_points(_N);

		if (input_distrib == "Moment") generateCyclicPolytope(vector_of_points, _D);
		else if (input_distrib == "Grid") generateUniformRandomGrid(vector_of_points, _N, _D);
		else if (input_distrib == "Circle") generateCircle(vector_of_points, _D);
		else if (input_distrib == "Sine") generateSineCurve(vector_of_points, _D);
		else { input_distrib = "Uniform"; generateBoxRandom(vector_of_points, _D);}

		vector<myVec *>  input(_N);
		for (size_t i = 0; i < _N; i++) input[i] = &vector_of_points[i];


		if (_THRESHOLD == 0) _THRESHOLD = (_uindex)(0.5f * _N);

		vector<myEdge *>  output;

		cout << input_distrib << ", n: " << _N << ", d: " << _D << endl;

		
				cout << "Started.\n";
				crossing_number = computeMatching(input, output, timetaken);
				cout << "\t Finished within  " << timetaken << ", crossing number: " <<  crossing_number << endl;

				ofstream fout("outputdata.txt", ofstream::app);
				fout <<  input_distrib << ", input size: " <<  _N << ", dim: " << _D << ", time: " << timetaken  << ", crossing nr: "<< crossing_number << endl;
				fout.close();

				



		if (output_tofile) {
			string pointsfile = "p-n" + to_string(_N) + "d" + to_string(_D)  + input_distrib + ".txt";
			saveFile(vector_of_points, pointsfile);
			string edgesfile = "e-n" + to_string(_N) + "d" + to_string(_D)  + input_distrib + ".txt";
			saveFile(output, edgesfile);
		}

		
}





void saveFile(vector<myVec> & points, string & fname)
{
	ofstream fout(fname);
	for (size_t i = 0; i < points.size(); i++)
	{
		for (size_t j = 0; j < points[i].coordinates.size(); j++)
			fout << points[i].coordinates[j] << " ";
		fout << endl;
	}
	fout.close();
}


void saveFile(vector<myEdge *> & output, string & fname)
{
	ofstream fout(fname);
	for (size_t i = 0; i < output.size(); i++)
	{
		for (size_t j = 0; j < output[i]->p1->coordinates.size(); j++)
			fout << output[i]->p1->coordinates[j] << " ";
		fout << endl;
		for (size_t j = 0; j < output[i]->p2->coordinates.size(); j++)
			fout << output[i]->p2->coordinates[j] << " ";
		fout << endl << endl;
	}
	fout.close();
}






void generateBoxRandom(vector<myVec> & vector_of_points, _uindex _D)
{
	for (size_t i = 0; i < vector_of_points.size(); i++)
		for (size_t j = 0; j < _D; j++) vector_of_points[i].coordinates.push_back(rand() / ((_float)RAND_MAX));
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


void generateCircle(vector<myVec> & vector_of_points, _uindex _D)
{
	assert(_D == 2);
	size_t nby4 = (size_t)((_float)vector_of_points.size() / 4.0f);
	for (size_t i = 0; i < nby4; i++) {
		_float angle = (3.14f / 2.0f * i) / (_float)nby4;
		vector_of_points[i].coordinates.push_back(sin(angle));
		vector_of_points[i].coordinates.push_back(cos(angle));

		vector_of_points[i + nby4].coordinates.push_back(-sin(angle));
		vector_of_points[i + nby4].coordinates.push_back(cos(angle));

		vector_of_points[i + 2 * nby4].coordinates.push_back(sin(angle));
		vector_of_points[i + 2 * nby4].coordinates.push_back(-cos(angle));

		vector_of_points[i + 3 * nby4].coordinates.push_back(-sin(angle));
		vector_of_points[i + 3 * nby4].coordinates.push_back(-cos(angle));
	}
}

void generateSineCurve(vector<myVec> & vector_of_points, _uindex _D)
{
	assert(_D == 2);
	size_t nby4 = (size_t)((_float)vector_of_points.size() / 40.0f);
	for (size_t i = 0; i < vector_of_points.size(); i++) {
		_float angle = (3.14f / 2.0f * i) / (_float)nby4;
		vector_of_points[i].coordinates.push_back((float) 800.0f * i / vector_of_points.size());
		vector_of_points[i].coordinates.push_back(sin(angle));
	}
}
