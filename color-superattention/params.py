# Checkpoint dir
sub_dir = 'super_attent_v1_res_conection'
checkpoint_dir = ('./checkpoints/' + sub_dir)
checkpoint_dir_ema = ('./checkpoints/' + sub_dir + '_ema')
model_dir = ('./models/' + sub_dir)
model_dir_ema = ('./models/' + sub_dir + '_ema')

# Carbon Tracker settings
carbon_tracker_log_dir = ('./trace/log_carbon_tracker' + sub_dir)
monitor_epochs = -1  # Number of epoch to monitor (-1 for all epoch)
log_verbose = 1  # Verbosity level
components = 'gpu'  # Components to track

# Save images in tensorboard
save_val_img = 200
save_train_img = 500
save_model_train = 25000
