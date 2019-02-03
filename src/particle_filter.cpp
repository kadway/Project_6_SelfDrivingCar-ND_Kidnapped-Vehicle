/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */

   //return if already initialized
  if (is_initialized)
    return;

  num_particles = 50;

  std::default_random_engine gen;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

 Particle new_particle;
 
 for(int i=0; i<num_particles; i++){
   new_particle.id=i+1;
   new_particle.x = dist_x(gen);
   new_particle.y = dist_y(gen);
   new_particle.theta = dist_theta(gen);
   new_particle.weight = 1.0;
   particles.push_back(new_particle);
 }

 // Flag, filter is initialized
 is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  std::default_random_engine gen;
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  
  for(int i=0; i<num_particles; i++){

    //temp variables to make code easier to read
    double x0 = particles[i].x;
    double y0 = particles[i].y;
    double angle = particles[i].theta;
    if(fabs(yaw_rate) > 0.00001){ //yaw rate not zero
      // velocity / yaw_rate
      double v_yaw = velocity / yaw_rate;
      // yaw_rate * dt
      double yaw_dt = yaw_rate * delta_t;
      // sin(theta + yaw_rate * dt)
      double sin_angle_yaw = sin(angle + yaw_dt);
      // cos(theta + yaw_rate * dt)
      double cos_angle_yaw = cos(angle + yaw_dt);
      //calculate the predition and add random gaussian noise
      particles[i].x = x0 + (v_yaw * (sin_angle_yaw - sin(angle)) ) + dist_x(gen);
      particles[i].y = y0 + (v_yaw * (cos(angle) - cos_angle_yaw) ) + dist_y(gen);
      particles[i].theta = angle + yaw_dt + dist_theta(gen);
    }
    else{
     //calculate the prediction and add random gaussian noise
     particles[i].x = x0 + velocity * delta_t * cos(angle)+ dist_x(gen);
     particles[i].y = y0 + velocity * delta_t * sin(angle) + dist_y(gen);
     
    }

  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations, double sensor_range) {
    //run through all observations..
   for(int obsNum=0; obsNum<(int)observations.size(); obsNum++) {
      
      //set min distance for association as the sensor range itself
      double min_distance = sensor_range;
      //temp variables for x and y pos of the observation
      double x_obs = observations[obsNum].x;
      double y_obs = observations[obsNum].y;

      //for each observation run through the landmarks which are whitin sensor range of the particle
      for(int predNum=0; predNum<(int)predicted.size(); predNum++){
          //temp variables for x, y and id of the landmark
          double x_pred = predicted[predNum].x;
          double y_pred = predicted[predNum].y;
          double id_pred = predicted[predNum].id;
         
          //calculate the distance between landmark and observation
          double distance = dist(x_obs, y_obs, x_pred, y_pred);
          if(distance < min_distance){
            min_distance = distance;
            observations[obsNum].id = id_pred;
          }
      } 
   } 
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
   // go over each particle
   for (int particleNum=0; particleNum<num_particles; particleNum++){
    
     double particle_x = particles[particleNum].x;
     double particle_y = particles[particleNum].y;
     double particle_angle = particles[particleNum].theta;
     particles[particleNum].weight=1;
 
     // aux vector for converted observations to map coordinates
     vector <LandmarkObs> observations_map;
     //go over each observation and convert to map coordinates relative to that of the particle position
     for(int obsNum=0; obsNum< (int)observations.size(); obsNum++){
       // transform to map x coordinates
       double x_map = particle_x + (cos(particle_angle) * observations[obsNum].x) - (sin(particle_angle) * observations[obsNum].y);
       // transform to map y coordinate
       double y_map = particle_y + (sin(particle_angle) * observations[obsNum].x) + (cos(particle_angle) * observations[obsNum].y);
       observations_map.push_back(LandmarkObs{observations[obsNum].id,x_map,y_map});

     }// for converted observations to map coordinates

     // vector to hold the landmarks that are whithin sensor range
     vector<LandmarkObs> in_range;
     //check which landmarks are within sensor range from the particle in question
     for(int lmNum=0; lmNum<(int)map_landmarks.landmark_list.size(); lmNum++){
       //temp for landmark x,y,id
       double landm_x = map_landmarks.landmark_list[lmNum].x_f;
       double landm_y = map_landmarks.landmark_list[lmNum].y_f;
       // (id - 1) because the ids on the map_data.txt start in 1 and not 0
       int landm_id = map_landmarks.landmark_list[lmNum].id_i - 1 ;

       //calculate the Euclidean distance between particle and landmarks
       // if distance is within sensor range then save the landmark
       double dist_range = dist(particle_x, particle_y, landm_x, landm_y);
       
       if (sensor_range > dist_range){
         in_range.push_back(LandmarkObs{landm_id,landm_x,landm_y});
       }
     }//for landmarks in sensor range

     // Associate observations to their closest landmarks which are whithin sensor range distance from the particle
     dataAssociation(in_range, observations_map, sensor_range);

     //loop through the observations again and take the respective associated landmark position to update the particle weight
     double weight_new = 1.0;
     for(unsigned int obsNum=0; obsNum < observations_map.size(); obsNum++){
     
       double x_obs = observations_map[obsNum].x;
       double y_obs = observations_map[obsNum].y;
       int lm_id = observations_map[obsNum].id;

       double mu_x = map_landmarks.landmark_list[lm_id].x_f;
       double mu_y = map_landmarks.landmark_list[lm_id].y_f;

       double sig_x = std_landmark[0];
       double sig_y = std_landmark[1];
       
       // update the particle weight using Multivariate-Gaussian PDF
       double gauss_norm = 1.0 / (2 * M_PI * sig_x * sig_y);
       // calculate exponent
       double exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2))) + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
       // calculate weight using normalization terms and exponent and do the product with previous weights
       weight_new = gauss_norm * exp(-exponent);
       //avoid a zero weight particle in the cases where the probability is zero for at least one pair of observation/landmark
       if(weight_new == 0){
         particles[particleNum].weight *= 0.0000000000001;
       }
       else{
       	particles[particleNum].weight *= weight_new;
       }
     }//for associated observations
     
     //save the new particle weight in the weights vector for use in resample step
     weights.push_back(particles[particleNum].weight);

   }//for particles
}


void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

   // vector to hold new particles
   std::vector<Particle> new_particles;

  // get the sum of the weights for normalization
  double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
  //normalize the weights
  for(int i=0; i<(int)weights.size();i++){
    weights[i]=weights[i]/sum_weights;
  }

  //Initialize the random number generator and uniform distribuitions necessary for the resampling algorithm
  std::default_random_engine gen;
  std::uniform_int_distribution<int> int_dist(0,num_particles-1);
  double max_weight=*max_element(weights.begin(), weights.end());
  std::uniform_real_distribution<double> beta_dist(0,max_weight);
  //init the variables
  int index = int_dist(gen);
  double beta = 0;

  
  // using the resampling wheel to resample the particles
  for (int particleNum=0; particleNum<num_particles; particleNum++){
    beta += beta_dist(gen) * 2 * max_weight;
    while (beta > weights[index]){
        beta -= weights[index];
        index = (index +1) % num_particles;
    }
    //save selected particle to new vector
    new_particles.push_back(particles[index]);
  }
  //update the particles to the new resampled vector of particles
  particles = new_particles;
  //clear the weights vector for next run
  weights.clear();
}




void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
