/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

#define DEBUG_VERBOSE false

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// set the number of particles
	num_particles = 100;  // less particles, run quicker, 

	if (DEBUG_VERBOSE) {
		num_particles = 1;  // less particles, run quicker, 
	}

	default_random_engine gen;
	normal_distribution<double> N_x(x, std[0]);
	normal_distribution<double> N_y(y, std[1]);
	normal_distribution<double> N_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++) {
		Particle	particle;
		particle.id = i;
		particle.x = N_x(gen);
		particle.y = N_y(gen);
		particle.theta = N_theta(gen);
		particle.weight = 1.0;

		particles.push_back(particle);
		weights.push_back(1.0);

		if (DEBUG_VERBOSE) {
			cout << "initial particle:" << particle.x << ", " << particle.y << ", " << particle.theta << endl;
		}
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine	gen;

	for (int i = 0; i < num_particles; i++) {
		double	new_x, new_y, new_theta = 0.0;
		
		if (fabs(yaw_rate) < 0.0001) {
			double vdt = velocity * delta_t; // velocity over time 
			new_x = particles[i].x + vdt * cos(particles[i].theta);
			new_y = particles[i].y + vdt * sin(particles[i].theta);
			new_theta = particles[i].theta;
		}
		else {
			double vyr = velocity / yaw_rate;  
			double yrdt = yaw_rate * delta_t; // yaw_rate over time
			new_x = particles[i].x + (vyr) *   (sin(particles[i].theta + yrdt) - sin(particles[i].theta));
			new_y = particles[i].y + (vyr) *   (cos(particles[i].theta) - cos(particles[i].theta + yrdt));
			new_theta = particles[i].theta + yrdt;
		}

		// add some noise to help spawn new particles
		// When adding noise, use std::normal_distribution and std::default_random_engine.
		normal_distribution<double> N_x(new_x, std_pos[0]);
		normal_distribution<double> N_y(new_y, std_pos[1]);
		normal_distribution<double> N_theta(new_theta, std_pos[2]);

		particles[i].x = N_x(gen);
		particles[i].y = N_y(gen);
		particles[i].theta = N_theta(gen);

		if (DEBUG_VERBOSE) {
			cout << "--------------------- Prediction ------------------------" << endl;
			cout << "velocity:" << velocity << "; yaw_rate:" << yaw_rate << endl;
			cout << "prediction:" << particles[i].x << " , " << particles[i].y << " , " << particles[i].theta << endl;
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	if (DEBUG_VERBOSE) {
		cout << "--------------------- Associations ------------------------" << endl;
	}
	
	for (int i = 0; i < observations.size(); i++){
		double min_distance = numeric_limits<double>::max();
		LandmarkObs currentObs = observations[i];
		int landmark_id = -1;

		for (int j = 0; j < predicted.size(); j++) {
			LandmarkObs currentPred = predicted[j];
			double current_distance = dist(currentObs.x, currentObs.y, currentPred.x, currentPred.y);

			if (current_distance < min_distance) {
				min_distance = current_distance;
				landmark_id = j;
				observations[i].id = currentPred.id;
			}
		}

		if (DEBUG_VERBOSE) {
			cout << "lmi: " << observations[i].id << " observations" << "(" << observations[i].x << ", " << observations[i].y << ")  ";
			cout << "predicted " << "(" << predicted[landmark_id].x << ", " << predicted[landmark_id].y << ") Dist: " << min_distance << endl;
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0; i < particles.size(); i++) {
		vector<int>				associations;
		vector<double>			sense_x;
		vector<double>			sense_y;
		vector<LandmarkObs>		trans_observations;

		double cos_theta = cos(particles[i].theta);
		double sin_theta = sin(particles[i].theta);

		// Initial particle weight to 1;
		particles[i].weight = 1.0;

		// Transform observations into map coordinates
		if (DEBUG_VERBOSE) {
			cout << "--------------------- Transformations -----------------------" << endl;
		}
		LandmarkObs current_obs;
		for (int j = 0; j < observations.size(); j++) {
			current_obs = observations[j];

			// Homogenous Transformation
			LandmarkObs	trans_obs;
			trans_obs.x = particles[i].x + (current_obs.x * cos_theta) - (current_obs.y * sin_theta);
			trans_obs.y = particles[i].y + (current_obs.x * sin_theta) + (current_obs.y * cos_theta);
			trans_obs.id = current_obs.id;

			if (DEBUG_VERBOSE) {
				cout << "observations(" << observations[j].x << ", " << observations[j].y << ") ";
				cout << "--> trans_obs(" << trans_obs.x << ", " << trans_obs.y << ")" << endl;
			}

			trans_observations.push_back(trans_obs);
		}

		// Filter landmark based on distance between particle and landmark
		// Should be less than sensor_range
		vector<LandmarkObs>		landmarks_in_range;

		for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
			double lm_x = map_landmarks.landmark_list[k].x_f;
			double lm_y = map_landmarks.landmark_list[k].y_f;
			double distance = dist(particles[i].x, particles[i].y, lm_x, lm_y);
			if (distance < sensor_range) {
				landmarks_in_range.push_back(LandmarkObs{ map_landmarks.landmark_list[k].id_i,
					map_landmarks.landmark_list[k].x_f,
					map_landmarks.landmark_list[k].y_f });
			}
		}
		dataAssociation(landmarks_in_range, trans_observations);

		if (DEBUG_VERBOSE) {
			cout<<"-------------------- Weights -----------------------"<<endl;
		}
		
		for (int l = 0; l < trans_observations.size(); l++) {
			double lm_x, lm_y;
			int lm_id;

			for (int m = 0; m < landmarks_in_range.size(); m++) {
				if (trans_observations[l].id == landmarks_in_range[m].id) {
					lm_id = landmarks_in_range[m].id;
					break;
				}
			}

			double meas_x = trans_observations[l].x;
			double meas_y = trans_observations[l].y;
			double mu_x = map_landmarks.landmark_list[lm_id - 1].x_f;
			double mu_y = map_landmarks.landmark_list[lm_id - 1].y_f;
				

			//multivariate Gaussian probability
			long double gauss_norm = (1 / (2 * M_PI * std_landmark[0] * std_landmark[1]));
			long double exponent = ((pow(meas_x - mu_x, 2) / (2 * pow(std_landmark[0], 2))) + (pow(meas_y - mu_y, 2) / (2 * pow(std_landmark[1], 2))));
			long double prob = gauss_norm * exp(-exponent);
			if (DEBUG_VERBOSE) {
				cout << "lmi: " << lm_id << " landmark(" << mu_x << ", " << mu_y << ") trans_observation(" << trans_observations[l].x << ", " << trans_observations[l].y << ") prob: " << prob << endl;
			}
			if (prob > 0.0) {
				particles[i].weight *= prob;
			}
			
			associations.push_back(lm_id);
			sense_x.push_back(trans_observations[l].x);
			sense_y.push_back(trans_observations[l].y);
		}

		particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);
		weights[i] = particles[i].weight;
		if (DEBUG_VERBOSE) {
			cout << "Weight[" << i << "]=" << weights[i] << endl;
		}
	}
}


void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	default_random_engine gen;
	discrete_distribution<int> distribution(weights.begin(), weights.end());
	vector<Particle> resample_particles;

	for (int i = 0; i < num_particles; i++) {
		resample_particles.push_back(particles[distribution(gen)]);
	}

	particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
