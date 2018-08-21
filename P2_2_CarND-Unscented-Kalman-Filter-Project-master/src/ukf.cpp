#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>
#define EPS 0.001
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  is_initialized_=false;

  n_x_=5;
  n_aug_=7;
  lambda_=3-n_aug_;
  Xsig_pred_=MatrixXd(n_x_,2*n_aug_+1);
  weights_=VectorXd(2*n_aug_+1);
  NIS_radar_=0.0;
  NIS_laser_=0.0;
  
  H_laser_=MatrixXd(2,5);
  H_laser_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;
    R_laser_=MatrixXd(2,2);
    R_laser_<<0.0225,0,
    0,0.0225;

  
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */


void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if ((meas_package.sensor_type_==MeasurementPackage::RADAR &&use_radar_)||
      (meas_package.sensor_type_==MeasurementPackage::LASER &&use_laser_)){
    
    if(!is_initialized_){
      x_<<1,1,1,1,1;
      P_<<1,0,0,0,0,
          0,1,0,0,0,
          0,0,1,0,0,
          0,0,0,1,0,
          0,0,0,0,1;
      
      
      if(meas_package.sensor_type_==MeasurementPackage::LASER && use_laser_){
        x_ << meas_package.raw_measurements_(0),meas_package.raw_measurements_(1), 0, 0, 0;
          if (fabs(x_(0))<EPS and fabs(x_(1))<EPS ){
              x_(0)=EPS;
              x_(1)=EPS;
          }
      }
      else if (meas_package.sensor_type_==MeasurementPackage::RADAR && use_radar_){
        float rho=meas_package.raw_measurements_(0);
        float phi=meas_package.raw_measurements_(1);
        float rho_dot=meas_package.raw_measurements_(2);
        x_(0)=rho* cos(phi);
        x_(1)=rho* sin(phi);
        float vx=rho_dot*cos(phi);
        float vy=rho_dot*sin(phi);
        x_(2)=sqrt(vx*vx+vy*vy);
        x_(3)=0;
        x_(4)=0;}
      weights_(0)=lambda_/(lambda_+n_aug_);
        for (int i=1;i<weights_.size();i++){
            weights_(i)=0.5/(n_aug_+lambda_);
        }
      time_us_=meas_package.timestamp_;
  
      is_initialized_=true;
      return;}
      
    double dt=(meas_package.timestamp_-time_us_)/1000000.0;
    time_us_=meas_package.timestamp_;
    
    
    Prediction(dt);
    
    if(meas_package.sensor_type_==MeasurementPackage::LASER){
      UpdateLidar(meas_package);}
    else if(meas_package.sensor_type_==MeasurementPackage::RADAR){
      UpdateRadar(meas_package);}
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  MatrixXd Xsig=MatrixXd(n_x_,2*n_x_+1);
  MatrixXd A=P_.llt().matrixL();
  //lambda_=3-n_aug_;
  Xsig.col(0)=x_;
  
  for(int i=0;i<n_x_;i++){
    Xsig.col(i+1)=x_+sqrt(lambda_+n_aug_)*A.col(i);
    Xsig.col(i+1+n_x_)=x_-sqrt(lambda_+n_aug_)*A.col(i);}
  
  VectorXd x_aug=VectorXd(n_aug_);
  MatrixXd P_aug=MatrixXd(n_aug_,n_aug_);
  MatrixXd Xsig_aug=MatrixXd(n_aug_,2*n_aug_+1);
  //lambda_=3-n_aug_;
    
  x_aug.fill(0.0);
  
  x_aug.head(5)=x_;

  
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5)=P_;
  P_aug(5,5)=std_a_*std_a_;
  P_aug(6,6)=std_yawdd_*std_yawdd_;
  
  MatrixXd L=P_aug.llt().matrixL();
  
  Xsig_aug.col(0)=x_aug;
  for (int i=0;i<n_aug_;i++){
    Xsig_aug.col(i+1)=x_aug+sqrt(lambda_+n_aug_)*L.col(i);
    Xsig_aug.col(i+1+n_aug_)=x_aug-sqrt(lambda_+n_aug_)*L.col(i);}
  
  for(int i=0;i<2*n_aug_+1;i++){
    double p_x=Xsig_aug(0,i);
    double p_y=Xsig_aug(1,i);
    double v=  Xsig_aug(2,i);
    double yaw=Xsig_aug(3,i);
    double yawd=Xsig_aug(4,i);
    double nu_a=Xsig_aug(5,i);
    double nu_yawdd=Xsig_aug(6,i);
    
    double px_p,py_p;
    
    if(fabs(yawd)>EPS){
      px_p=p_x+v/yawd*(sin(yaw+yawd*delta_t)-sin(yaw));
      py_p=p_y+v/yawd*(cos(yaw)-cos(yaw+yawd*delta_t));}
    else{
      px_p=p_x+v*delta_t*cos(yaw);
      py_p=p_y+v*delta_t*sin(yaw);}
    
    double v_p=v;
    double yaw_p=yaw+yawd*delta_t;
    double yawd_p=yawd;
    
    px_p=px_p+0.5*nu_a*delta_t*delta_t*cos(yaw);
    py_p=py_p+0.5*nu_a*delta_t*delta_t*sin(yaw);
    v_p=v_p+nu_a*delta_t;
    
    yaw_p=yaw_p+0.5*nu_yawdd*delta_t*delta_t;
    yawd_p=yawd_p+nu_yawdd*delta_t;
    
    Xsig_pred_(0,i)=px_p;
    Xsig_pred_(1,i)=py_p;
    Xsig_pred_(2,i)=v_p;
    
    Xsig_pred_(3,i)=yaw_p;
    
    
    Xsig_pred_(4,i)=yawd_p;}
    


  
/*
  x_.fill(0.0);
  for (int i=0;i<2*n_aug_+1;i++){
    x_=x_+weights_(i)*Xsig_pred_.col(i);}*/
    
  x_=Xsig_pred_*weights_;
  
  P_.fill(0.0);
  for (int i=0;i<2*n_aug_+1;i++){
    VectorXd x_diff=Xsig_pred_.col(i)-x_;
    //while(x_diff(3)>M_PI) x_diff(3)-=2.*M_PI;
    //while(x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    P_=P_+weights_(i)*x_diff*x_diff.transpose();}

    
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
    /*
  VectorXd z=meas_package.raw_measurements_;
  int n_z=2;
  MatrixXd Zsig=MatrixXd(n_z, 2*n_aug_+1);
  for (int i=0;i<2*n_aug_+1;i++){
    double p_x=Xsig_pred_(0,i);
    double p_y=Xsig_pred_(1,i);
    
    Zsig(0,i)=p_x;
    Zsig(1,i)=p_y;}
  
  VectorXd z_pred=VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0;i>2*n_aug_+1;i++){
    z_pred=z_pred+weights_(i)*Zsig.col(i);}
  
  MatrixXd S=MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i=0;i<2*n_aug_+1;i++){
    VectorXd z_diff=Zsig.col(i)-z_pred;
    S=S+weights_(i)*z_diff*z_diff.transpose();}
  
  MatrixXd R=MatrixXd(n_z,n_z);
  R<<std_laspx_*std_laspx_,0,
    0,std_laspy_*std_laspy_;
  S=S+R;
  
  MatrixXd Tc=MatrixXd(n_x_,n_z);
  Tc.fill(0.0);
  for(int i=0;i<2*n_aug_+1;i++){
    VectorXd z_diff=Zsig.col(i)-z_pred;
    while (z_diff(1)>M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    VectorXd x_diff=Xsig_pred_.col(i)-x_;
    while(x_diff(3)>M_PI) x_diff(3)-=2.*M_PI;
    while(x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    Tc=Tc+weights_(i)*x_diff*z_diff.transpose();}
  
  MatrixXd K=Tc*S.inverse();
  VectorXd z_diff=z-z_pred;
  NIS_laser_=z_diff.transpose()*S.inverse()*z_diff;
  
  x_=x_+K*z_diff;
  P_=P_-K*S*K.transpose();  */
    
    VectorXd z = meas_package.raw_measurements_;
    VectorXd z_pred = H_laser_ * x_;
    VectorXd y = z - z_pred;
    MatrixXd H_t = H_laser_.transpose();
    MatrixXd S = H_laser_ * P_ * H_t+R_laser_;
    MatrixXd S_inv = S.inverse();
    MatrixXd K = P_ * H_t * S_inv;
    
    //new estimate and NIS
    x_ += K * y;
    P_ -= K * H_laser_ * P_;
    NIS_laser_ = y.transpose() * S_inv * y;
     
  
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  VectorXd z=meas_package.raw_measurements_;
  int n_z=3;
  MatrixXd Zsig=MatrixXd(n_z,2*n_aug_+1);
  Zsig.fill(0.0);

  
  for(int i=0;i<2*n_aug_+1;i++){
    double p_x=Xsig_pred_(0,i);
    double p_y=Xsig_pred_(1,i);
    double v=Xsig_pred_(2,i);
    double yaw=Xsig_pred_(3,i);
    while (yaw>M_PI) yaw-=2.*M_PI;
    while (yaw<-M_PI) yaw+=2.*M_PI;
    
    double v1=cos(yaw)*v;
    double v2=sin(yaw)*v;
      
    /*if(fabs(p_x)<EPS and fabs(p_y)<EPS){
          Zsig(0,i)=0;
          Zsig(1,i)=0;
          Zsig(2,i)=0;
      }
    else{*/
    
  if(fabs(p_x)<0.0001) p_x+=0.0001;
  Zsig(0,i)=sqrt(p_x*p_x+p_y*p_y);
  Zsig(1,i)=atan2(p_y,p_x);
  Zsig(2,i)=(p_x*v1+p_y*v2)/sqrt(p_x*p_x+p_y*p_y);}
  
  VectorXd z_pred=VectorXd(n_z);
  z_pred.fill(0.0);
  for(int i=0;i<2*n_aug_+1;i++){
    z_pred=z_pred+weights_(i)*Zsig.col(i);}
  
  MatrixXd S=MatrixXd(n_z,n_z);
  S.fill(0.0);
  for(int i=0;i<2*n_aug_+1;i++){
    VectorXd z_diff=Zsig.col(i)-z_pred;
    while (z_diff(1)>M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    S=S+weights_(i)*z_diff*z_diff.transpose();}
  
  MatrixXd R=MatrixXd(n_z,n_z);
  R<<std_radr_*std_radr_,0,0,
     0,std_radphi_*std_radphi_,0,
     0,0,std_radrd_*std_radrd_;
  S=S+R;
  
  MatrixXd Tc=MatrixXd(n_x_,n_z);
  Tc.fill(0.0);
  for(int i=0;i<2*n_aug_+1;i++){
    VectorXd z_diff=Zsig.col(i)-z_pred;
    while( z_diff(1)>M_PI) z_diff(1)-=2.*M_PI;
    while( z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    
    VectorXd x_diff=Xsig_pred_.col(i)-x_;
    while(x_diff(3)>M_PI) x_diff(3)-=2.*M_PI;
    while(x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    
    Tc=Tc+weights_(i)*x_diff*z_diff.transpose();}
  
  MatrixXd K=Tc*S.inverse();
  VectorXd z_diff=z-z_pred;
  
  while(z_diff(1)>M_PI) z_diff(1)-=2.*M_PI;
  while(z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
  
  NIS_radar_=z_diff.transpose()*S.inverse()*z_diff;
  x_=x_+K*z_diff;
  P_=P_-K*S*K.transpose();
}
