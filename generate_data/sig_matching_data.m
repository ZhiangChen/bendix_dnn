function simulation = sig_matching_data(example_num, display_figures)
% display_figures = 0: not display figures
% display_figures = 1: display figures
%file to generate vehicle-tracking synthetic data
% start w/ vehicle dynamics:
%
%    dist += vrel*dt
%    vrel+= acc*dt
  %  acc = Ka*(dist_nom - dist) + gaussian_noise
    
 %   lat_offset: measured + to the left, - to the right; nominally quanta of a lane width
 %   lat_offset += v_lat*dt
  %  v_lat + a_lat*dt  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<******************************** v_lat += a_lat*dt
 %  a_lat = Ka*(nom_lat_offset - lat_offset) + noise
  %   nom_lat_offset is ramp fnc to various lane offsets;
      
    %vehicle width is width_true
    %equiv angles: theta_min, theta_max, where:
   % dist*tan(theta_min) = lat_offset - width/2
  %  dist*tan(theta_max) = lat_offset + width/2
  
 % example_num = '8'; %change this for each run; will append to fnames
   width=0.5*randn+2 %actual width of back of vehicle object
   lane_num = round(2*rand)-1%integer lane number: +1 to left, -1 to right
   change_lane_num = round(2*rand)-1

   
   lane_width = 5.0; %5m lane width
   latpos_des = lane_num*lane_width; %desired same lane; change by quanta of lane_width
  
  %set params:
  dt=0.04; %sim time step
  t_final = 30 %100.0; %final time of sim
  t_lane_change = t_final*rand
  
  %motion noise
  std_acc_noise = 1; % 0.1g random accel noise
  std_acc_lat = 1; % lateral acceleration noise
  
  %perception noise
  std_fwd_pos_sensor_noise = 0.2 % std forward/following position sensor, in meters <<<<<<<<<<<<<<<<<<<<***************** 0.2 << 0.5
  std_lat_pos_sensor_noise = 0.5 % sensor noise for lateral position, in meters
  std_vel_sensor_noise = 1 % std m/sec vel sensor noise
  std_width_sensor_noise = 0.25 %std of width sensor, in meters
  
  std_ang_sensor_noise = 0.05 %angular radians noise
  std_cam_fwd_pos_noise = 4 %stdev of camera-based distance estimate
  std_cam_pos_bias = 2 %camera-based position estimate bias
  std_cam_vel_noise = 1 %stdev of camera-based vrel

  %vehicle-following control params (driver behavior)
  dbrake = 10; %open gap if too close
  dfar=100.0;  %close gap if too far out
  dfollow = 30.0;  %nominal following distance
  Kacc_brake = 1; %this much fdbk if exceed brake/far boundaries
  Kacc_follow = 0; % zero gain-> random drift of following distance in brake/far range
  Kvel_follow = 0.05; %velocity feedback--tries to keep velocity steady-ish
  
    %lateral veh position feedback controls (driver behavior)
  Kacc_lat = 0.2;
  Kvel_lat = 0.5;
  
  %initialize state variables:
  a = 0; %acc of following; not really a state variable, but want to log it
  vrel=0.0;
  dist = dfollow; % start at nom following distance
  latpos = latpos_des; %lateral offset
  latvel = 0.0; %lateral velocity
 
 %formula to convert fwd/lat distances to angles at left/right of bumper
  %dist*tan(theta_min) = latpos-width/2
  theta_min = atan2(latpos-width/2, dist);
  theta_max = atan2(latpos+width/2,dist);
  %initialize storage vectors for plotting/saving results
  vrel_vec = [vrel];
  dist_vec = [dist];
  width_vec = [width];
  latpos_vec = [latpos];
  t_vec=[0];
  acc_vec=[a];
  theta_min_vec = [theta_min];
  theta_max_vec = [theta_max];

%start the simulation
  for t=dt:dt:t_final
   if ( t>t_lane_change)
      latpos_des = change_lane_num*lane_width;	
   end
   a =  std_acc_noise*randn; %like Brownian motion--noisy accel, zero mean
   a_lat = std_acc_lat*randn; %same for lateral accel
   
   %add driver feedback:
   a_lat=a_lat +Kacc_lat*(latpos_des-latpos)-Kvel_lat*latvel;
   %Euler one-step integration
   latvel+= a_lat*dt;
   latpos+=latvel*dt;

  %driver feedback for following distance
   if (dist< dbrake)  %if getting too close, react w/ higher accel
   	a+=Kacc_brake*(dbrake-dist)-Kvel_follow*vrel;
   	%end
  elseif  (dist>dfar) %if gap gets to big, data will be useless, so close the gap
   	a+=(dfar-dist)*Kacc_brake -Kvel_follow*vrel;
   	%end
   else %approx following distance..mostly keeping relative velocity near zero
   	a+=(dfollow-dist)*Kacc_follow-Kvel_follow*vrel;
   	end
   
   	
   vrel+= a*dt;  %Euler one-step integration for dynamics of following distance
   dist+= vrel*dt;
   
   %convert right/left corners of bumper to angles, based on width, following dist and lat offset
     theta_min = atan2(latpos-width/2, dist);
    theta_max = atan2(latpos+width/2,dist);
    
    %store simu data in vectors
      theta_min_vec = [theta_min_vec,theta_min];
    theta_max_vec = [theta_max_vec,theta_max];
      acc_vec=[acc_vec,a];
     vrel_vec = [vrel_vec,vrel];
     t_vec=[t_vec,t];
     dist_vec = [dist_vec,dist];
   
     latpos_vec = [latpos_vec,latpos];
   width_vec = [width_vec,width]; %boring/perfect
   %plot out the simu results	
 end



  %add sensor noise to all simu variables:
  [dummy,npts] = size(t_vec);
  %these 4 are virtual radar measurements
   sensed_dist_vec = std_fwd_pos_sensor_noise*randn(1,npts)+dist_vec;
   sensed_vrel_vec = std_vel_sensor_noise*randn(1,npts)+vrel_vec;
   sensed_width_vec  = std_width_sensor_noise*randn(1,npts)+width_vec;
   sensed_latpos_vec = std_lat_pos_sensor_noise*randn(1,npts)+latpos_vec;
   %these 4 are virtual camera measurements
   sensed_theta_max_vec = std_ang_sensor_noise*randn(1,npts)+theta_max_vec;
   sensed_theta_min_vec = std_ang_sensor_noise*randn(1,npts)+theta_min_vec;  
   sensed_cam_dist_vec = std_cam_fwd_pos_noise*randn(1,npts) +dist_vec +std_cam_pos_bias*ones(1,npts);
   sensed_cam_vrel_vec = std_cam_vel_noise*randn(1,npts)+vrel_vec;
   mean_width = mean(sensed_width_vec)

if display_figures == true
   figure(1)
  plot(t_vec,20*vrel_vec,'b',t_vec,dist_vec,'r',t_vec,acc_vec,'g',t_vec,10*latpos_vec,'k')
  xlabel('time (sec)')
  ylabel('m, m/sec')
  title('distance (red) and 20x vrel (blue), 10x lat pos (blk)')

   figure(2)
   plot(t_vec,20*sensed_vrel_vec,'b',t_vec,sensed_dist_vec,'r',t_vec,10*sensed_latpos_vec,'k',t_vec,sensed_width_vec,'g')
   title('sensed vars: dist (r), 20x vrel (b), 10x latpos (k), width (g)')
   xlabel('time (sec)')
  figure(3)
  plot(t_vec,sensed_theta_min_vec,'r',t_vec,sensed_theta_max_vec,'b')
  xlabel('time (sec)')
  ylabel ('ang (rad)')
  title('sensed min theta (r) and max theta (b)')
  
  figure(4)
     plot(t_vec,20*sensed_cam_vrel_vec,'b',t_vec,sensed_cam_dist_vec,'r')
   title('sensed camera vars: dist (r), 20x vrel (b)')
   xlabel('time (sec)')
end

   %save the data separately for radar/camera; name each simu file by example number
  radar_vals = [sensed_dist_vec; sensed_vrel_vec; sensed_latpos_vec; sensed_width_vec];
  cam_vals = [sensed_cam_dist_vec; sensed_cam_vrel_vec; sensed_theta_min_vec; sensed_theta_max_vec];
  
  %save -ascii radar_vals_4 radar_vals
  %save -ascii cam_vals_4 cam_vals
     fname_radar = 'radar_vals_';
   fname_radar_exmpl = [fname_radar,num2str(example_num)]
   fname_cam = 'cam_vals_';
   fname_cam_exmpl=[fname_cam,num2str(example_num)]
   radar_vals_trans = radar_vals';
   cam_vals_trans = cam_vals';
   simulation.radar_vals_trans = radar_vals_trans
   simulation.cam_vals_trans = cam_vals_trans
   simulation.dt = dt
   %save as 4 features per row, each row is another sample (easier to read text file in editor)
   save( fname_radar_exmpl, '-ascii', 'radar_vals_trans'); 
   save( fname_cam_exmpl, '-ascii', 'cam_vals_trans'); 
