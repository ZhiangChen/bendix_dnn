function snippet = match_sigs_create_training_snippets(display_figures)

load cam_vals_1;
load cam_vals_2;
load radar_vals_1;
load radar_vals_2;
load radar_vals_3;
load radar_vals_4;

Npts_old = 20;
Npts_new= 5;
sample_length = Npts_old+Npts_new;
N_good_examples = 100;
N_bad_examples = 100;
%matrix_good_examples = zeros(N_good_examples,sample_length);
matrix_good_examples = [];
%matrix_bad_examples = zeros(N_bad_examples,sample_length);
matrix_bad_examples = [];
[nsamps,ntracks] = size(radar_vals_1)
tvals = 1:nsamps;
%all radar_vals tracks have the same number of samples
%create GOOD training data...each series is Npts_old + Npts_new long
%get some data from radar_vals_1:

for i_example_good=1:25
	nstart = randi(nsamps - sample_length);
	matrix_good_examples = [matrix_good_examples;radar_vals_1(nstart:nstart+sample_length-1,1)'];
	matrix_good_examples = [matrix_good_examples;radar_vals_2(nstart:nstart+sample_length-1,1)'];
	matrix_good_examples = [matrix_good_examples;radar_vals_3(nstart:nstart+sample_length-1,1)'];
	matrix_good_examples = [matrix_good_examples;radar_vals_4(nstart:nstart+sample_length-1,1)'];
end
[n_good_examples,n_samps_training] = size(matrix_good_examples)

%now lets make some bad data:
for i_example_good=1:25
	nstart = randi(nsamps - Npts_old);
	track = radar_vals_1(nstart:nstart+Npts_old-1);
	nstart = randi(nsamps - Npts_new);
	track = [track,radar_vals_2(nstart:nstart+Npts_new-1,1)'];
	matrix_bad_examples = [matrix_bad_examples;track];
	
	nstart = randi(nsamps - Npts_old);
	track = radar_vals_2(nstart:nstart+Npts_old-1);
	nstart = randi(nsamps - Npts_new);
	track = [track,radar_vals_3(nstart:nstart+Npts_new-1,1)'];
	matrix_bad_examples = [matrix_bad_examples;track];	
	
	nstart = randi(nsamps - Npts_old);
	track = radar_vals_3(nstart:nstart+Npts_old-1);
	nstart = randi(nsamps - Npts_new);
	track = [track,radar_vals_4(nstart:nstart+Npts_new-1,1)'];
	matrix_bad_examples = [matrix_bad_examples;track];	
	
	nstart = randi(nsamps - Npts_old);
	track = radar_vals_4(nstart:nstart+Npts_old-1);
	nstart = randi(nsamps - Npts_new);
	track = [track,radar_vals_1(nstart:nstart+Npts_new-1,1)'];
	matrix_bad_examples = [matrix_bad_examples;track];				
	
end

snippet.good_example = matrix_good_examples
snippet.bad_example = matrix_bad_examples

if display_figures == true
	[n_bad_examples,n_samps_training] = size(matrix_bad_examples)
	tvals = 1:n_samps_training
  	i = randi(n_bad_examples)
	figure(1)
	subplot(2,1,1)
	plot(tvals,matrix_good_examples(i,:))
	title('good')
	subplot(2,1,2)
	plot(tvals,matrix_bad_examples(i,:))
	title('bad')
%{
	figure(2)
	subplot(2,1,1)
	plot(tvals,cam_vals_1(:,1),tvals,radar_vals_1(:,1))
	subplot(2,1,2)
	plot(tvals,cam_vals_1(:,2),tvals,radar_vals_1(:,2))
	%subplot(4,1,3)
	%plot(tvals,cam_vals_1(:,3),tvals,radar_vals_1(:,3))
	%subplot(4,1,4)
	%plot(tvals,cam_vals_1(:,4),tvals,radar_vals_1(:,4))

	figure(3)
	subplot(2,1,1)
	plot(tvals,cam_vals_2(:,1),tvals,radar_vals_2(:,1))
	subplot(2,1,2)
	plot(tvals,cam_vals_2(:,2),tvals,radar_vals_2(:,2))

	dist_diff11 = cam_vals_1(:,1)-radar_vals_1(:,1);
	dist_diff12 = cam_vals_1(:,1)-radar_vals_2(:,2);
	figure(4)
	plot(tvals,dist_diff11,tvals,dist_diff12)
%}
end

