# from carbontracker.tracker import CarbonTracker
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage
from utils import save_checkpoint, imagenet_norm
from params import sub_dir, checkpoint_dir, checkpoint_dir_ema, model_dir

import time

def training_color(loader_train, loader_val,
                model_color, num_epochs, device, optimizer,
                loss_feat, lpips, mode):
    # Save images in tensorboard
    save_val_img = int(len(loader_val)/10)
    save_train_img = int(len(loader_train)/10)
    save_model_train = int(len(loader_train)/2) - 1
    # Initialize carbon tracker
    # tracker = CarbonTracker(epochs=(num_epochs // len(loader_train) + 1),
    #                             log_dir=carbon_tracker_log_dir,
    #                             monitor_epochs=monitor_epochs,
    #                             verbose=log_verbose,
    #                             components=components)

    # Writer will output to ./runs/ directory by default
    tb = SummaryWriter('./runs/' + sub_dir + '/', flush_secs=2)

    n_total_steps = len(loader_train)
    n_total_steps_val = len(loader_val)

    val_iter = int(len(loader_train) / 2) - 1
    val_avg = 0

    alpha = 20
    alpha_lpips = 0.15
    ema = ExponentialMovingAverage(model_color.parameters(), decay=0.995)

    for epoch in range(num_epochs):
        model_color.train()
        # tracker.epoch_start()
        epoch += 1
        best_loss = 100.0
        loss_val_epoch = 0.0
        running_loss = 0.0
        running_loss_l2 = 0.0
        running_loss_hist = 0.0
        running_loss_lpips = 0.0
        epoch_start_time = time.time()

        if epoch == 1:
            sum_idx = 0
        else:
            sum_idx += len(loader_train)
            running_loss = 0.0
            running_loss_l2 = 0.0
            running_loss_lpips = 0.0

        for idx, (img_rgb_target, img_target_gray, img_target_ab,
                ref_rgb, ref_gray, target_slic, ref_slic_all, img_ref_ab,
                img_gray_map, gray_real, ref_real) in enumerate(loader_train):

            model_color.train()
            step_start_time = time.time()

            # Target data
            img_gray_map = (img_gray_map).to(device=device, dtype=torch.float)
            img_target_gray = (img_target_gray).to(
                device=device, dtype=torch.float)
            img_rgb_target = img_rgb_target.to(
                device=device, dtype=torch.float)
            gray_real = gray_real.to(device=device, dtype=torch.float)
            target_slic = target_slic

            # Loading references
            ref_rgb_torch = ref_rgb.to(device=device, dtype=torch.float)
            img_ref_gray = (ref_gray).to(device=device, dtype=torch.float)
            ref_slic_all = ref_slic_all

            # VGG19 normalization
            img_ref_rgb_norm = imagenet_norm(ref_rgb_torch, device)

            # VGG19 normalization
            img_target_gray_norm = img_target_gray
            img_ref_gray_norm = img_ref_gray

            # Clearing the optimizer
            optimizer.zero_grad()

            ab_pred, _, pred_rgb_torch = model_color(img_target_gray_norm,
                                                                  img_ref_gray_norm,
                                                                  img_target_gray,
                                                                  target_slic,
                                                                  ref_slic_all,
                                                                  img_gray_map,
                                                                  gray_real,
                                                                  img_ref_rgb_norm,
                                                                  device, mode)

            # Loss calculation
            l2_loss = loss_feat(ab_pred, img_target_ab)

            # Lpips loss: inputs should be between [-1, 1]

            lpips_loss = lpips(((pred_rgb_torch * 2.0) - 1.0),
                               ((img_rgb_target.to(device=device, dtype=torch.float) * 2.0) - 1.0))

            # Total loss calculation
            loss = alpha * l2_loss + alpha_lpips * lpips_loss
            running_loss_l2 = running_loss_l2 + l2_loss.item()

            running_loss_lpips = running_loss_lpips + lpips_loss.item()
            running_loss = running_loss + loss.item()

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            # ema step
            ema.update()

            if (idx % 10 == 0):
                print("Epoch[{}/{}]({}/{}): Loss_train: {:.4f} Loss_l2: {:.4f}".format(
                    epoch, num_epochs, idx, n_total_steps),
                    " Loss_lpips: {:.4f} Step time: {} seconds".format(
                    loss.item(), alpha * l2_loss.item(),  alpha_lpips * lpips_loss.item(),
                    time.time() - step_start_time))

            # Checkpoint

            if (idx + sum_idx > 10000) and (loss.item() < best_loss):
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model_color.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                is_best = True
                best_loss = loss.item()
                save_checkpoint(checkpoint, is_best,
                                checkpoint_dir, (idx + sum_idx), model_dir)
                print('Best model updated')

            if (idx + sum_idx > 10000) and (loss.item() < best_loss):
                checkpoint_ema = {
                    'epoch': epoch + 1,
                    'state_dict': ema.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                is_best = True
                best_loss = loss.item()
                save_checkpoint(checkpoint_ema, is_best,
                                checkpoint_dir_ema, (idx + sum_idx), model_dir)
                print('Best model  ema updated')

            if (idx % save_model_train == 0):
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model_color.state_dict(),
                    'optimizer': optimizer.state_dict()
                }

                is_best = False
                save_checkpoint(checkpoint, is_best,
                                checkpoint_dir, (idx + sum_idx), model_dir)
                print('Last model saved')

            if (idx % save_model_train == 0):
                checkpoint_ema = {
                    'epoch': epoch + 1,
                    'state_dict': ema.state_dict(),
                    'optimizer': optimizer.state_dict()
                }

                is_best = False
                save_checkpoint(checkpoint_ema, is_best,
                                checkpoint_dir_ema, (idx + sum_idx), model_dir)
                print('Last model saved')

            # Write some images in tensorboard every save_train_img step

            if (idx % save_train_img == 0):
                img_grid = torchvision.utils.make_grid(img_target_gray)
                img_rgb_pred = torchvision.utils.make_grid(pred_rgb_torch)
                img_target_rgb_view = torchvision.utils.make_grid(
                    img_rgb_target)
                img_ref_rgb_view = torchvision.utils.make_grid(ref_rgb_torch)
                tb.add_image('Target image in step', img_grid, idx + sum_idx)
                tb.add_image('Reference image in RGB',
                             img_ref_rgb_view, idx + sum_idx)
                tb.add_image('predicted image in step',
                             img_rgb_pred, idx + sum_idx)
                tb.add_image('Target RGB image in step',
                             img_target_rgb_view, idx + sum_idx)
                tb.flush()

            # Writing losses in tensorboard
            tb.add_scalar("Total loss per step", loss, idx + sum_idx)
            tb.add_scalar("L2 loss per step", alpha * l2_loss, idx + sum_idx)
            tb.add_scalar("LPIPS loss per step", alpha_lpips *
                          lpips_loss, idx + sum_idx)

            # Forcing refreshing on tensorboard
            tb.flush()

            # # Evaluation step
            if (idx > 0) and (idx % val_iter == 0):
                print('Entering validation block')

                with torch.no_grad():

                    model_color.eval()

                    val_avg += 1

                    if val_avg == 1:
                        sum_idx_val = 0
                        loss_avg_val = 0
                        loss_avg_val_l2 = 0

                    else:
                        sum_idx_val += len(loader_val)
                        loss_avg_val = 0
                        loss_avg_val_l2 = 0

                    for idx_val, (val_img_rgb_target, val_img_target_gray,
                                val_img_target_ab, val_ref_rgb, val_ref_gray,
                                val_target_slic, val_ref_all, val_img_ref_ab,
                                val_img_gray_map, val_gray_real, val_ref_real) in enumerate(loader_val):

                        step_val_start_time = time.time()
                        val_img_gray_map = (val_img_gray_map).to(
                            device=device, dtype=torch.float)
                        val_gray_real = val_gray_real.to(
                            device=device, dtype=torch.float)

                        val_img_target_gray = val_img_target_gray.to(
                            device=device, dtype=torch.float)
                        val_img_rgb_target_data = val_img_rgb_target.to(
                            device=device, dtype=torch.float)

                        val_ref_gray = val_ref_gray.to(
                            device=device, dtype=torch.float)
                        val_img_rgb_ref = val_ref_rgb.to(
                            device=device, dtype=torch.float)
                        val_ref_all = val_ref_all
                        val_target_slic = val_target_slic

                        # #VGG19 normalization

                        val_img_ref_color_norm = imagenet_norm(
                            val_img_rgb_ref, device)

                        val_img_target_gray_norm = val_img_target_gray
                        val_img_ref_gray_norm = val_ref_gray

                        val_ab_pred, _, pred_val_rgb_torch = model_color(
                            val_img_target_gray_norm, val_img_ref_gray_norm, val_img_target_gray,
                            val_target_slic, val_ref_all, val_img_gray_map,
                            val_gray_real, val_img_ref_color_norm, device)

                        loss_l2_val = loss_feat(val_ab_pred, val_img_target_ab)

                        # Lpips loss: inputs should be between [-1, 1]
                        loss_lpips_val = lpips(((pred_val_rgb_torch * 2.0) - 1.0), (
                            (val_img_rgb_target_data.to(device=device, dtype=torch.float) * 2.0) - 1.0))

                        loss_val = alpha * loss_l2_val + alpha_lpips * loss_lpips_val
                        loss_avg_val_l2 = alpha * loss_avg_val_l2 + loss_l2_val.item()
                        loss_avg_val = loss_avg_val + loss_val.item()  # Acumulating loss

                        # Write some images in tensorboard every 1 step
                        if (idx_val % save_val_img == 0):
                            val_img_grid = torchvision.utils.make_grid(
                                val_img_target_gray)
                            val_img_rgb_pred = torchvision.utils.make_grid(
                                pred_val_rgb_torch)
                            val_img_target_rgb_view = torchvision.utils.make_grid(
                                val_img_rgb_target_data)
                            val_img_ref_rgb_view = torchvision.utils.make_grid(
                                val_img_rgb_ref)
                            tb.add_image('Target image in validation step',
                                         val_img_grid, idx_val + sum_idx_val)
                            tb.add_image(
                                'predicted image in validation step', val_img_rgb_pred, idx_val + sum_idx_val)
                            tb.add_image('Target RGB image in validation step',
                                         val_img_target_rgb_view, idx_val + sum_idx_val)
                            tb.add_image('Reference image RGB in validation',
                                         val_img_ref_rgb_view, idx_val + sum_idx_val)
                            tb.flush()

                        tb.add_scalar("Loss_val per step",
                                      loss_val, idx_val + sum_idx_val)
                        tb.add_scalar("L2 loss_val per step",
                                      loss_l2_val, idx_val + sum_idx_val)
                        tb.add_scalar("LPIPS loss_val per step",
                                      loss_lpips_val, idx_val + sum_idx_val)
                        print("Epoch_val[{}/{}]({}/{}): Loss_val: {:.4f} Step time: {} seconds".format(
                            epoch, num_epochs, idx_val, n_total_steps_val,
                            loss_val.item(), time.time() - step_val_start_time))

                    print('Validation finished')

                    avg_loss_val = loss_avg_val / len(loader_val)
                    tb.add_scalar("Loss_val avg", avg_loss_val, val_avg)
                    tb.flush()
                    loss_val_epoch = loss_val_epoch + avg_loss_val

        # Average loss for each epoch
        epoch_val = loss_val_epoch / val_iter
        epoch_loss = running_loss / len(loader_train)
        epoch_loss_l2 = running_loss_l2 / len(loader_train)
        epoch_loss_hist = running_loss_hist / len(loader_train)
        tb.add_scalar("Total loss per epoch", epoch_loss, epoch)
        tb.add_scalar("L2 loss per epoch", epoch_loss_l2, epoch)
        tb.add_scalar("Hist loss per epoch", epoch_loss_hist, epoch)
        tb.add_scalar("Validation Loss per epoch",  epoch_val, epoch)
        print("Epoch[{}/{}]({}/{}): Loss: {:.4f} Epoch time{} seconds".format(
            epoch, num_epochs, idx,
            n_total_steps,
            epoch_loss,
            time.time() - epoch_start_time))
    # tracker.epoch_end()

    # Save final model
    torch.save(model_color.state_dict(), './models')
    print('Training saved and finished')
    tb.close()
    # tracker.stop()
